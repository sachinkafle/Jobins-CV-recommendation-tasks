"""CV parsing with LangGraph"""
from typing import Optional, List, Dict
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END

from .models import ResumeSchema
from ..utils import config, logger

# --- State Definition ---

class CVParserState(TypedDict):
    """State for CV parsing graph"""
    pdf_path: str
    raw_text: str
    parsed_resume: Optional[ResumeSchema]
    errors: Optional[List[str]]
    attempts: int

# --- Nodes ---

def load_pdf_node(state: CVParserState) -> Dict:
    """Extract text from PDF"""
    logger.info(f"Loading PDF: {state['pdf_path']}")
    try:
        loader = PyPDFLoader(state['pdf_path'])
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])
        return {"raw_text": full_text, "errors": []}
    except Exception as e:
        logger.error(f"PDF loading error: {e}")
        return {"errors": [str(e)]}

def parse_resume_node(state: CVParserState) -> Dict:
    """Parse resume using LLM"""
    logger.info(f"Parsing resume (Attempt {state.get('attempts', 0) + 1})")

    llm = ChatOpenAI(model=config.llm_model, temperature=0)
    structured_llm = llm.with_structured_output(ResumeSchema)

    system_msg = """You are an expert CV parser. Extract information from the resume text 
    into the strictly defined JSON Resume schema. 
    - If dates are ambiguous, output them as YYYY-MM.
    - Map 'Company' to 'name' in the work section.
    - Ensure lists are extracted fully.
    - Extract all skills comprehensively."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "Resume Text:\n{text}")
    ])

    chain = prompt | structured_llm

    try:
        result = chain.invoke({"text": state["raw_text"]})
        return {"parsed_resume": result, "attempts": state.get("attempts", 0) + 1}
    except Exception as e:
        logger.error(f"Parsing error: {e}")
        return {"errors": [str(e)], "attempts": state.get("attempts", 0) + 1}

def validation_node(state: CVParserState) -> Dict:
    """Validate critical fields"""
    logger.info("Validating parsed data")
    data = state["parsed_resume"]

    if not data or not data.basics.name:
        return {"errors": ["Critical: Name missing"]}

    return {"errors": []}

# --- Router ---

def router(state: CVParserState) -> str:
    """Decide retry or finish"""
    if state.get("errors") and state["attempts"] < 3:
        logger.warning(f"Validation failed, retrying: {state['errors']}")
        return "parse_resume"
    return END

# --- Graph Construction ---

def build_cv_parser_graph():
    """Build and compile CV parser graph"""
    workflow = StateGraph(CVParserState)

    workflow.add_node("load_pdf", load_pdf_node)
    workflow.add_node("parse_resume", parse_resume_node)
    workflow.add_node("validate", validation_node)

    workflow.set_entry_point("load_pdf")
    workflow.add_edge("load_pdf", "parse_resume")
    workflow.add_edge("parse_resume", "validate")
    workflow.add_conditional_edges(
        "validate",
        router,
        {
            "parse_resume": "parse_resume",
            END: END
        }
    )

    return workflow.compile()

# --- Public API ---

class CVParser:
    """CV Parser API"""

    def __init__(self):
        self.graph = build_cv_parser_graph()

    def parse(self, pdf_path: str) -> Optional[ResumeSchema]:
        """Parse a CV from PDF"""
        initial_state = {
            "pdf_path": pdf_path,
            "attempts": 0,
            "errors": []
        }

        final_state = self.graph.invoke(initial_state)

        if final_state.get("parsed_resume"):
            logger.info(f"Successfully parsed CV: {pdf_path}")
            return final_state["parsed_resume"]
        else:
            logger.error(f"Failed to parse CV: {final_state.get('errors')}")
            return None
