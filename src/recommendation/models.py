"""Recommendation models"""
from typing import List
from pydantic import BaseModel, Field


class JobPosting(BaseModel):
    """Job posting information"""
    job_id: str
    title: str
    company: str
    description: str
    requirements: str


class MatchingFactors(BaseModel):
    """Detailed matching factors with scores"""
    skills_match: float = Field(..., ge=0.0, le=1.0, description="Technical skills alignment")
    experience_match: float = Field(..., ge=0.0, le=1.0, description="Years and role relevance")
    education_match: float = Field(..., ge=0.0, le=1.0, description="Degree level match")
    semantic_similarity: float = Field(..., ge=0.0, le=1.0, description="Overall semantic similarity")


class JobRecommendation(BaseModel):
    """Single job recommendation"""
    job_id: str
    job_title: str
    company: str
    match_score: float = Field(..., ge=0.0, le=1.0, description="Overall weighted match score")
    matching_factors: MatchingFactors
    matched_skills: List[str] = Field(default_factory=list, description="Skills that match job requirements")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills candidate lacks")
    explanation: str = Field(..., description="Human-readable match explanation")


class RecommendationOutput(BaseModel):
    """Complete recommendation output"""
    candidate_id: str
    candidate_name: str
    recommendations: List[JobRecommendation]
