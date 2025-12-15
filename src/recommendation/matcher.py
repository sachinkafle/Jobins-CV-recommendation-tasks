"""Job matching engine with LangGraph"""
from typing import List, Dict
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .models import JobPosting, MatchingFactors, JobRecommendation, RecommendationOutput
from .database import db_manager
from ..parsers.models import ResumeSchema
from ..utils import config, logger


# --- State Definition ---

class MatchingState(TypedDict):
    """State for matching graph"""
    candidate_profile: str
    candidate_id: str
    candidate_name: str
    retrieved_jobs: List[Dict]  # Changed to Dict to include similarity scores
    ranked_results: List[JobRecommendation]


# --- LLM Output Schema ---

class MatchAnalysis(BaseModel):
    """LLM output for single job analysis"""
    skills_match: float = Field(..., ge=0.0, le=1.0)
    experience_match: float = Field(..., ge=0.0, le=1.0)
    education_match: float = Field(..., ge=0.0, le=1.0)
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    explanation: str


# --- Nodes ---

def retrieve_jobs_node(state: MatchingState) -> Dict:
    """Retrieve relevant jobs using vector search with HNSW"""
    logger.info(f"Retrieving jobs for {state['candidate_id']}")
    
    # Semantic search - returns jobs WITH similarity scores
    results = db_manager.search_similar_jobs(state["candidate_profile"], k=10)
    
    # Keep raw results with similarity scores
    # Each result has: job_id, title, company, requirements, content, similarity_score
    logger.info(f"Retrieved {len(results)} jobs from HNSW vector store")
    
    if results:
        logger.info(f"Top match similarity: {results[0].get('similarity_score', 0):.3f}")
        logger.info(f"Lowest match similarity: {results[-1].get('similarity_score', 0):.3f}")
    
    return {"retrieved_jobs": results}


def rank_jobs_node(state: MatchingState) -> Dict:
    """Rank and grade jobs using LLM + HNSW similarity scores"""
    logger.info("Ranking retrieved jobs with LLM analysis")
    
    llm = ChatOpenAI(model=config.llm_model, temperature=0)
    structured_llm = llm.with_structured_output(MatchAnalysis)
    
    system_msg = """You are an expert HR AI. Compare the candidate profile to the job description.

Calculate specific match scores (0.0 to 1.0):
- skills_match: How well candidate's technical skills align with job requirements (0.0 = no overlap, 1.0 = perfect match)
- experience_match: How well years of experience and role relevance match (0.0 = no match, 1.0 = perfect match)
- education_match: How well degree level and field match requirements (0.0 = no match, 1.0 = perfect match)

List specific skills:
- matched_skills: Skills the candidate has that match job requirements (list of strings)
- missing_skills: Required skills the candidate lacks (list of strings)

Provide explanation:
- A 1-2 sentence summary of why this is/isn't a good match, highlighting key strengths"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", """Candidate Profile:
{candidate}

Job Title: {job_title}
Company: {company}

Job Description & Requirements:
{job_reqs}

Analyze the match and provide scores, matched/missing skills, and explanation.""")
    ])
    
    chain = prompt | structured_llm
    results = []
    weights = config.matching_weights
    
    for job_data in state["retrieved_jobs"]:
        try:
            # Extract similarity score from HNSW vector search
            semantic_similarity = job_data.get('similarity_score', 0.0)
            
            # Get LLM analysis
            analysis = chain.invoke({
                "candidate": state["candidate_profile"],
                "job_title": job_data.get('title', 'Unknown'),
                "company": job_data.get('company', 'Unknown'),
                "job_reqs": job_data.get('requirements', '') + "\n\n" + job_data.get('content', '')
            })
            
            # Calculate weighted final score using ACTUAL HNSW similarity
            final_score = (
                analysis.skills_match * weights.get("skills", 0.4) +
                analysis.experience_match * weights.get("experience", 0.3) +
                analysis.education_match * weights.get("education", 0.2) +
                semantic_similarity * weights.get("semantic", 0.1)  # Using REAL score from HNSW
            )
            
            # Create matching factors
            factors = MatchingFactors(
                skills_match=round(analysis.skills_match, 2),
                experience_match=round(analysis.experience_match, 2),
                education_match=round(analysis.education_match, 2),
                semantic_similarity=round(semantic_similarity, 2)  # REAL HNSW score
            )
            
            # Create recommendation
            results.append(JobRecommendation(
                job_id=job_data.get("job_id", "unknown"),
                job_title=job_data.get("title", "Unknown Title"),
                company=job_data.get("company", "Unknown Company"),
                match_score=round(final_score, 2),
                matching_factors=factors,
                matched_skills=analysis.matched_skills,
                missing_skills=analysis.missing_skills,
                explanation=analysis.explanation
            ))
            
            logger.info(
                f"Matched {job_data.get('job_id')}: "
                f"HNSW={semantic_similarity:.2f}, "
                f"Skills={analysis.skills_match:.2f}, "
                f"Final={final_score:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error grading job {job_data.get('job_id')}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Sort by score descending
    results.sort(key=lambda x: x.match_score, reverse=True)
    
    logger.info(f"Ranked {len(results)} jobs. Top score: {results[0].match_score if results else 0}")
    
    return {"ranked_results": results}


# --- Graph Construction ---

def build_matching_graph():
    """Build matching workflow graph"""
    workflow = StateGraph(MatchingState)
    
    workflow.add_node("retrieve", retrieve_jobs_node)
    workflow.add_node("rank", rank_jobs_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rank")
    workflow.add_edge("rank", END)
    
    return workflow.compile()


# --- Public API ---

class JobMatcher:
    """Job matching API"""
    
    def __init__(self):
        self.graph = build_matching_graph()
    
    def match(self, resume: ResumeSchema, candidate_id: str) -> RecommendationOutput:
        """
        Match a resume to jobs and return structured recommendations
        
        Args:
            resume: Parsed resume in JSON Resume schema
            candidate_id: Unique identifier for candidate
            
        Returns:
            RecommendationOutput with ranked job recommendations
        """
        # Create candidate profile summary
        skills_str = ', '.join([
            ', '.join(skill.keywords) if skill.keywords else skill.name 
            for skill in resume.skills
        ])
        
        work_str = ', '.join([
            f"{w.position} at {w.name} ({w.startDate or 'N/A'} - {w.endDate or 'Present'})" 
            for w in resume.work
        ])
        
        education_str = ', '.join([
            f"{e.studyType or 'Degree'} in {e.area or 'N/A'} from {e.institution}" 
            for e in resume.education
        ])
        
        profile = f"""
Name: {resume.basics.name}
Email: {resume.basics.email or 'N/A'}
Professional Summary: {resume.basics.summary or 'N/A'}

Skills: {skills_str}

Work Experience:
{work_str}

Education:
{education_str}

Projects: {len(resume.projects)} projects
Certifications: {', '.join([c.name for c in resume.certificates]) if resume.certificates else 'None'}
Languages: {', '.join([lang.language for lang in resume.languages]) if resume.languages else 'N/A'}
"""
        
        initial_state = {
            "candidate_id": candidate_id,
            "candidate_name": resume.basics.name,
            "candidate_profile": profile.strip(),
            "retrieved_jobs": [],
            "ranked_results": []
        }
        
        logger.info(f"Starting job matching for {resume.basics.name} (ID: {candidate_id})")
        
        final_state = self.graph.invoke(initial_state)
        
        logger.info(
            f"Generated {len(final_state['ranked_results'])} recommendations for {candidate_id}. "
            f"Top match score: {final_state['ranked_results'][0].match_score if final_state['ranked_results'] else 0}"
        )
        
        # Return structured output
        return RecommendationOutput(
            candidate_id=candidate_id,
            candidate_name=resume.basics.name,
            recommendations=final_state["ranked_results"]
        )
