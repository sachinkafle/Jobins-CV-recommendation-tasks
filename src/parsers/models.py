"""Data models for CV parsing (JSON Resume Schema - Official)"""
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl

class Location(BaseModel):
    """Location information"""
    address: Optional[str] = Field(None, description="Street address")
    postalCode: Optional[str] = Field(None, description="Postal/ZIP code")
    city: Optional[str] = Field(None, description="City name")
    countryCode: Optional[str] = Field(None, description="ISO 3166-1 alpha-2 code")
    region: Optional[str] = Field(None, description="State or province")

class Profile(BaseModel):
    """Social media profile"""
    network: Optional[str] = Field(None, description="Social network name (e.g., Twitter, LinkedIn)")
    username: Optional[str] = Field(None, description="Username on the network")
    url: Optional[str] = Field(None, description="Profile URL")

class Basics(BaseModel):
    """Basic candidate information"""
    name: str = Field(..., description="Full name of the candidate")
    label: Optional[str] = Field(None, description="Job title/tagline (e.g., 'Software Engineer')")
    image: Optional[str] = Field(None, description="Profile image URL")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    url: Optional[str] = Field(None, description="Personal website URL")
    summary: Optional[str] = Field(None, description="Short professional bio (2-3 sentences)")
    location: Optional[Location] = Field(None, description="Candidate location")
    profiles: List[Profile] = Field(default_factory=list, description="Social media profiles")

class Work(BaseModel):
    """Work experience entry"""
    name: Optional[str] = Field(None, description="Company name")
    position: Optional[str] = Field(None, description="Job title")
    url: Optional[str] = Field(None, description="Company website")
    startDate: Optional[str] = Field(None, description="Start date (YYYY-MM-DD or YYYY-MM)")
    endDate: Optional[str] = Field(None, description="End date (YYYY-MM-DD, YYYY-MM, or 'Present')")
    summary: Optional[str] = Field(None, description="Overview of responsibilities")
    highlights: List[str] = Field(default_factory=list, description="Key achievements")

class Volunteer(BaseModel):
    """Volunteer experience"""
    organization: Optional[str] = Field(None, description="Organization name")
    position: Optional[str] = Field(None, description="Role/position")
    url: Optional[str] = Field(None, description="Organization website")
    startDate: Optional[str] = Field(None, description="Start date")
    endDate: Optional[str] = Field(None, description="End date")
    summary: Optional[str] = Field(None, description="Description of volunteer work")
    highlights: List[str] = Field(default_factory=list, description="Key contributions")

class Education(BaseModel):
    """Education entry"""
    institution: Optional[str] = Field(None, description="University or school name")
    url: Optional[str] = Field(None, description="Institution website")
    area: Optional[str] = Field(None, description="Major or field of study")
    studyType: Optional[str] = Field(None, description="Degree type (e.g., Bachelor, Master, PhD)")
    startDate: Optional[str] = Field(None, description="Start date")
    endDate: Optional[str] = Field(None, description="End date or 'Present'")
    score: Optional[str] = Field(None, description="GPA or grade (e.g., '3.8', 'First Class')")
    courses: List[str] = Field(default_factory=list, description="Notable courses")

class Award(BaseModel):
    """Award or honor"""
    title: Optional[str] = Field(None, description="Award title")
    date: Optional[str] = Field(None, description="Date received")
    awarder: Optional[str] = Field(None, description="Organization that gave the award")
    summary: Optional[str] = Field(None, description="Description of the award")

class Certificate(BaseModel):
    """Certification"""
    name: Optional[str] = Field(None, description="Certificate name")
    date: Optional[str] = Field(None, description="Date obtained")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    url: Optional[str] = Field(None, description="Certificate URL/verification link")

class Publication(BaseModel):
    """Publication"""
    name: Optional[str] = Field(None, description="Publication title")
    publisher: Optional[str] = Field(None, description="Publisher name")
    releaseDate: Optional[str] = Field(None, description="Publication date")
    url: Optional[str] = Field(None, description="Publication URL")
    summary: Optional[str] = Field(None, description="Short description")

class Skill(BaseModel):
    """Skill category"""
    name: Optional[str] = Field(None, description="Skill category (e.g., 'Web Development')")
    level: Optional[str] = Field(None, description="Proficiency level (e.g., 'Master', 'Intermediate')")
    keywords: List[str] = Field(default_factory=list, description="Specific skills")

class Language(BaseModel):
    """Language proficiency"""
    language: Optional[str] = Field(None, description="Language name")
    fluency: Optional[str] = Field(None, description="Proficiency level (e.g., 'Native', 'Fluent', 'Professional')")

class Interest(BaseModel):
    """Personal interest"""
    name: Optional[str] = Field(None, description="Interest category")
    keywords: List[str] = Field(default_factory=list, description="Specific interests")

class Reference(BaseModel):
    """Professional reference"""
    name: Optional[str] = Field(None, description="Reference person's name")
    reference: Optional[str] = Field(None, description="Reference statement/testimonial")

class Project(BaseModel):
    """Project entry"""
    name: Optional[str] = Field(None, description="Project name")
    startDate: Optional[str] = Field(None, description="Start date")
    endDate: Optional[str] = Field(None, description="End date or 'Present'")
    description: Optional[str] = Field(None, description="Project description")
    highlights: List[str] = Field(default_factory=list, description="Key achievements")
    url: Optional[str] = Field(None, description="Project URL")

class ResumeSchema(BaseModel):
    """Complete JSON Resume Schema (v1.0.0)"""
    basics: Basics
    work: List[Work] = Field(default_factory=list)
    volunteer: List[Volunteer] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    awards: List[Award] = Field(default_factory=list)
    certificates: List[Certificate] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    interests: List[Interest] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "basics": {
                    "name": "John Doe",
                    "label": "Software Engineer",
                    "email": "john@example.com",
                    "phone": "+1-234-567-8900",
                    "url": "https://johndoe.com",
                    "summary": "Experienced software engineer...",
                    "location": {
                        "city": "San Francisco",
                        "countryCode": "US",
                        "region": "California"
                    }
                },
                "work": [{
                    "name": "Tech Corp",
                    "position": "Senior Engineer",
                    "startDate": "2020-01",
                    "endDate": "Present"
                }]
            }
        }
