"""
Project Matcher for finding existing Daruka projects or generating hypothetical ones.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.vectorstore import ChromaManager


# Daruka capabilities for project generation
DARUKA_CAPABILITIES = """
Daruka.Earth Core Capabilities:
1. AI-Powered Biodiversity Monitoring - Species identification using bioacoustics and computer vision
2. Multimodal Data Integration - Satellite, drone, IoT sensors, and field data fusion
3. Real-Time MRV (Monitoring, Reporting, Verification) - Digital MRV for carbon and biodiversity credits
4. Community-Driven Data Collection - Mobile tools for local communities as data stewards
5. Carbon Credit Generation - Measurable carbon sequestration tracking
6. Biodiversity Credit Assessment - Ecosystem health and species abundance metrics
7. Climate Risk Modeling - Predictive analytics using ML and climate data

Technology Stack:
- Bioacoustic AI models (custom trained for regional species)
- Satellite imagery analysis (land cover, vegetation indices)
- IoT sensor networks (AudioMoth recorders, environmental sensors)
- Cloud analytics platform for real-time processing

Proven Track Record:
- India's first biodiversity credit project in Sundarbans
- 300+ local data stewards empowered
- 1000+ hours of bioacoustic data processed
- Partnerships with Cornell Lab of Ornithology, IISER Tirupati
"""


@dataclass
class ProjectMatch:
    """A matched or generated project."""
    name: str
    project_type: str  # "existing" or "generated"
    focus_areas: List[str]
    target_species: List[str]
    location: str
    description: str
    methodology: str
    expected_outcomes: List[str]
    relevance_score: float = 0.0
    source_chunk_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.project_type,
            "focus_areas": self.focus_areas,
            "target_species": self.target_species,
            "location": self.location,
            "description": self.description,
            "methodology": self.methodology,
            "expected_outcomes": self.expected_outcomes,
            "relevance_score": self.relevance_score
        }


class ProjectMatcher:
    """
    Matches grant requirements to existing Daruka projects,
    or generates hypothetical projects when no match exists.
    """
    
    PROJECTS_COLLECTION = "daruka_projects"
    MATCH_THRESHOLD = 0.6  # Score above this means "match found"
    
    def __init__(
        self,
        chroma_manager: ChromaManager,
        api_key: str,
        model: str = "claude-3-5-haiku-20241022"
    ):
        self.chroma = chroma_manager
        self.llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model=model,
            temperature=0.3  # Slightly creative for project generation
        )
    
    def find_matching_project(
        self,
        grant_focus: str,
        grant_requirements: str,
        top_k: int = 3
    ) -> Optional[ProjectMatch]:
        """
        Search for existing projects matching grant requirements.
        
        Args:
            grant_focus: Main focus of the grant (e.g., "raptor conservation")
            grant_requirements: Detailed requirements from grant
            top_k: Number of projects to consider
            
        Returns:
            ProjectMatch if found, None otherwise
        """
        # Combine focus and requirements for search
        search_query = f"{grant_focus}. {grant_requirements}"
        
        # Search existing projects
        results = self.chroma.search(
            query=search_query,
            collection_names=[self.PROJECTS_COLLECTION],
            top_k=top_k
        )
        
        if not results:
            return None
        
        # Check if best match exceeds threshold
        best_result = results[0]
        if best_result.score < self.MATCH_THRESHOLD:
            return None
        
        # Parse project from result
        try:
            metadata = best_result.metadata
            return ProjectMatch(
                name=metadata.get("project_name", "Unnamed Project"),
                project_type="existing",
                focus_areas=self._parse_list(metadata.get("focus_areas", "")),
                target_species=self._parse_list(metadata.get("target_species", "")),
                location=metadata.get("location", "India"),
                description=best_result.content,
                methodology=metadata.get("methodology", ""),
                expected_outcomes=self._parse_list(metadata.get("expected_outcomes", "")),
                relevance_score=best_result.score,
                source_chunk_id=best_result.chunk_id
            )
        except Exception as e:
            print(f"Error parsing project: {e}")
            return None
    
    def generate_hypothetical_project(
        self,
        grant_focus: str,
        grant_requirements: str,
        grant_context: str = ""
    ) -> ProjectMatch:
        """
        Generate a hypothetical project using LLM.
        
        Args:
            grant_focus: Main focus of the grant
            grant_requirements: Requirements from grant
            grant_context: Additional context about the grant
            
        Returns:
            Generated ProjectMatch
        """
        prompt = f"""You are helping Daruka.Earth create a project proposal for a conservation grant.

GRANT FOCUS: {grant_focus}

GRANT REQUIREMENTS:
{grant_requirements}

ADDITIONAL CONTEXT:
{grant_context}

DARUKA.EARTH CAPABILITIES:
{DARUKA_CAPABILITIES}

Generate a realistic, achievable project proposal that:
1. Directly addresses the grant's focus area
2. Uses Daruka's actual technology capabilities (bioacoustics, satellite, AI, community involvement)
3. Has measurable outcomes
4. Is achievable within 1-2 years
5. Includes locations relevant to the grant (if in India, suggest specific regions)

Output as JSON (no markdown, just pure JSON):
{{
    "project_name": "Creative but professional project name",
    "focus_areas": ["area1", "area2"],
    "target_species": ["species1", "species2"],
    "location": "Specific location in India or region",
    "description": "2-3 sentence project description",
    "methodology": "Brief methodology using Daruka's capabilities",
    "expected_outcomes": ["outcome1", "outcome2", "outcome3"]
}}"""

        messages = [
            SystemMessage(content="You are a conservation project designer. Output only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            # Parse JSON from response
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            project_data = json.loads(content)
            
            return ProjectMatch(
                name=project_data.get("project_name", "Generated Conservation Project"),
                project_type="generated",
                focus_areas=project_data.get("focus_areas", []),
                target_species=project_data.get("target_species", []),
                location=project_data.get("location", "India"),
                description=project_data.get("description", ""),
                methodology=project_data.get("methodology", ""),
                expected_outcomes=project_data.get("expected_outcomes", []),
                relevance_score=1.0  # Generated specifically for this grant
            )
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response: {response.content}")
            # Return a default project
            return ProjectMatch(
                name=f"Daruka Conservation Initiative - {grant_focus}",
                project_type="generated",
                focus_areas=[grant_focus],
                target_species=[],
                location="India",
                description=f"AI-powered conservation project focusing on {grant_focus} using Daruka.Earth's dMRV platform.",
                methodology="Bioacoustic monitoring, satellite imagery analysis, and community-driven data collection.",
                expected_outcomes=["Species population baseline", "Ecosystem health metrics", "Community engagement"],
                relevance_score=0.8
            )
    
    def get_or_generate_project(
        self,
        grant_focus: str,
        grant_requirements: str,
        grant_context: str = "",
        force_generate: bool = False
    ) -> ProjectMatch:
        """
        Get matching project or generate one if none found.
        
        Args:
            grant_focus: Main focus of the grant
            grant_requirements: Requirements from grant
            grant_context: Additional context
            force_generate: If True, always generate (skip search)
            
        Returns:
            ProjectMatch (existing or generated)
        """
        if not force_generate:
            # Try to find existing project
            existing = self.find_matching_project(grant_focus, grant_requirements)
            if existing:
                print(f"✅ Found matching project: {existing.name} (score: {existing.relevance_score:.2f})")
                return existing
        
        # Generate new project
        print(f"🔧 Generating hypothetical project for: {grant_focus}")
        return self.generate_hypothetical_project(grant_focus, grant_requirements, grant_context)
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse a comma-separated string or list."""
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
