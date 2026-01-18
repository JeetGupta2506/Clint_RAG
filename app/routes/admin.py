"""
Admin endpoints for system management.
"""

from fastapi import APIRouter, HTTPException, Depends

from app.config import get_settings, Settings
from app.models import StatsResponse, ClearResponse
from src.vectorstore import ChromaManager, EmbeddingsManager


router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats(settings: Settings = Depends(get_settings)):
    """
    Get system statistics.
    
    - Total documents and chunks
    - Collections available
    - Chunks per collection
    """
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        stats = chroma.get_all_stats()
        
        return StatsResponse(
            total_documents=stats["total_collections"],
            total_chunks=stats["total_chunks"],
            collections=stats["collections"],
            chunks_per_collection=stats["chunks_per_collection"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear", response_model=ClearResponse)
async def clear_database(
    confirm: bool = False,
    settings: Settings = Depends(get_settings)
):
    """
    Clear all data from ChromaDB.
    
    - Requires confirm=true parameter
    - Deletes all collections
    - Use with caution!
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=true to confirm deletion of all data"
        )
    
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        cleared = chroma.clear_all()
        
        return ClearResponse(
            collections_cleared=cleared,
            message=f"Cleared {len(cleared)} collections"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    confirm: bool = False,
    settings: Settings = Depends(get_settings)
):
    """
    Delete a specific collection.
    
    - collection_name: Name of collection to delete
    - confirm: Must be true to confirm deletion
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail=f"Add ?confirm=true to confirm deletion of collection '{collection_name}'"
        )
    
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        # Check if collection exists
        collections = chroma.list_collections()
        if collection_name not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found. Available: {collections}"
            )
        
        success = chroma.delete_collection(collection_name)
        
        if success:
            return {
                "message": f"Successfully deleted collection '{collection_name}'",
                "collection": collection_name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete collection")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections")
async def list_collections(settings: Settings = Depends(get_settings)):
    """List all available collections."""
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        collections = chroma.list_collections()
        
        return {
            "collections": collections,
            "count": len(collections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}")
async def view_collection(
    collection_name: str,
    limit: int = 10,
    settings: Settings = Depends(get_settings)
):
    """
    View chunks in a specific collection.
    
    - collection_name: Name of the collection to view
    - limit: Maximum number of chunks to return (default 10)
    """
    try:
        embeddings = EmbeddingsManager(
            model=settings.embedding_model
        )
        
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        collection = chroma.get_or_create_collection(collection_name)
        
        # Get all items from collection
        results = collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results and results["documents"]:
            for i, (doc, meta, chunk_id) in enumerate(zip(
                results["documents"], 
                results["metadatas"],
                results["ids"]
            )):
                chunks.append({
                    "id": chunk_id,
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "metadata": meta
                })
        
        return {
            "collection": collection_name,
            "total_in_collection": collection.count(),
            "showing": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/seed")
async def seed_sample_projects(
    settings: Settings = Depends(get_settings)
):
    """
    Seed the database with sample Daruka projects.
    Run this once to populate the projects collection.
    """
    from src.rag.project_matcher import ProjectMatcher
    
    try:
        embeddings = EmbeddingsManager(model=settings.embedding_model)
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        # Sample projects
        projects = [
            {
                "project_name": "Sundarbans Biodiversity Credit Project",
                "focus_areas": "mangroves, biodiversity credits, carbon, community conservation",
                "target_species": "birds, fish, crustaceans, mangrove species",
                "location": "Indian Sundarbans, West Bengal",
                "status": "active",
                "methodology": "AI-powered bioacoustic monitoring, satellite imagery analysis, community data stewards",
                "expected_outcomes": "300+ local data stewards, 1000+ hours bioacoustic data, measurable biodiversity credits",
                "content": """India's First Biodiversity Credit Project in the Sundarbans.
                
This flagship project in the Indian Sundarbans—one of the world's largest mangrove forests and a RAMSAR-recognized site—demonstrates Daruka.Earth's complete dMRV capabilities.

Key Achievements:
- Empowered 300+ local individuals (including forest dwellers and women) as data stewards
- Created green jobs through community-driven monitoring
- Processed 1000+ hours of bioacoustic data for species identification
- Piloted 500-hectare conservation zone
- Democratized climate finance by ensuring rural communities benefit directly

Technology: AudioMoth recorders, AI species identification, satellite imagery, mobile apps for community data collection."""
            },
            {
                "project_name": "BioGuardian: Real-Time Biodiversity Threat Detection Platform",
                "focus_areas": "AI, threat detection, climate resilience, ecosystem monitoring, multimodal",
                "target_species": "multi-species, amphibians, birds, mammals",
                "location": "Jharkhand, Sundarbans, India (scalable)",
                "status": "development",
                "methodology": "Multimodal AI using foundational models, bioacoustics, satellite, drone data fusion",
                "expected_outcomes": "Real-time threat alerts, ecosystem insights, automated MRV reporting, species trend analysis",
                "content": """BioGuardian: A Real-Time Biodiversity Threat Detection & Climate Resilience Platform

An AI-powered field intelligence platform that analyzes sound, satellite imagery, drone footage, and field reports in real-time.

Key Capabilities:
- Detect threats like illegal logging, species disappearance, or climate-induced degradation
- Generate ecosystem insights (species trends, rewilding opportunities)
- Deliver natural language responses to field teams and policymakers
- Automate reporting and MRV for biodiversity and climate projects

Technology: Gemini/Vertex AI, AutoML Vision, bioacoustic sensors, Earth Engine integration.
Timeline: 3-month accelerator readiness with pilots in Jharkhand and Sundarbans."""
            },
            {
                "project_name": "Western Ghats Avian Acoustic Monitoring",
                "focus_areas": "birds, raptors, acoustic monitoring, endemic species, rainforest conservation",
                "target_species": "raptors, eagles, kites, vultures, hornbills, endemic birds",
                "location": "Western Ghats, Karnataka and Kerala",
                "status": "planned",
                "methodology": "Dense AudioMoth network, AI species identification, community parabiologist program",
                "expected_outcomes": "Endemic species population baseline, habitat connectivity maps, community conservation network, 50+ trained parabiologists",
                "content": """Western Ghats Avian Acoustic Monitoring Project

Conservation initiative focusing on the Western Ghats—a UNESCO World Heritage Site and biodiversity hotspot.

Project Goals:
- Establish baseline population data for endemic and endangered bird species
- Monitor raptor populations including eagles, kites, and vultures
- Track hornbill abundance as indicator species for forest health
- Create acoustic fingerprint of healthy vs degraded forest patches

Methodology:
- Deploy 50+ AudioMoth recorders across altitude gradients
- Train AI models on Western Ghats-specific bird and raptor calls
- Partner with local communities as parabiologists
- Integrate satellite imagery for habitat mapping

Alignment: Supports State Forest Department mandates, India's Kunming-Montreal commitments, biodiversity credit potential."""
            }
        ]
        
        # Add to collection
        documents = [p["content"] for p in projects]
        metadatas = [{k: v for k, v in p.items() if k != "content"} for p in projects]
        ids = [f"project_{i}" for i in range(len(projects))]
        
        count = chroma.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            collection_name=ProjectMatcher.PROJECTS_COLLECTION
        )
        
        return {
            "message": f"Seeded {count} sample projects",
            "projects": [p["project_name"] for p in projects],
            "collection": ProjectMatcher.PROJECTS_COLLECTION
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel, Field
from typing import List, Optional


class ProjectInput(BaseModel):
    """Input model for adding a new project."""
    project_name: str = Field(..., description="Name of the project")
    description: str = Field(..., description="Detailed project description")
    focus_areas: List[str] = Field(..., description="List of focus areas e.g., ['raptors', 'conservation']")
    target_species: List[str] = Field(default=[], description="List of target species")
    location: str = Field(..., description="Geographic location")
    methodology: str = Field(default="", description="Project methodology")
    expected_outcomes: List[str] = Field(default=[], description="Expected outcomes")
    status: str = Field(default="planned", description="Status: active, planned, completed")


@router.post("/projects/add")
async def add_project(
    project: ProjectInput,
    settings: Settings = Depends(get_settings)
):
    """
    Add a new Daruka project to the database.
    
    This project will be searchable and used in future queries.
    """
    from src.rag.project_matcher import ProjectMatcher
    import uuid
    
    try:
        embeddings = EmbeddingsManager(model=settings.embedding_model)
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        # Create full content for embedding
        content = f"""{project.project_name}

{project.description}

Focus Areas: {', '.join(project.focus_areas)}
Target Species: {', '.join(project.target_species)}
Location: {project.location}
Methodology: {project.methodology}
Expected Outcomes: {', '.join(project.expected_outcomes)}
Status: {project.status}"""
        
        # Create metadata
        metadata = {
            "project_name": project.project_name,
            "focus_areas": ", ".join(project.focus_areas),
            "target_species": ", ".join(project.target_species),
            "location": project.location,
            "methodology": project.methodology,
            "expected_outcomes": ", ".join(project.expected_outcomes),
            "status": project.status
        }
        
        # Generate unique ID
        project_id = f"project_{uuid.uuid4().hex[:8]}"
        
        # Add to collection
        count = chroma.add_documents(
            documents=[content],
            metadatas=[metadata],
            ids=[project_id],
            collection_name=ProjectMatcher.PROJECTS_COLLECTION
        )
        
        return {
            "message": f"Project '{project.project_name}' added successfully",
            "project_id": project_id,
            "collection": ProjectMatcher.PROJECTS_COLLECTION
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects")
async def list_projects(
    settings: Settings = Depends(get_settings)
):
    """
    List all Daruka projects in the database.
    """
    from src.rag.project_matcher import ProjectMatcher
    
    try:
        embeddings = EmbeddingsManager(model=settings.embedding_model)
        chroma = ChromaManager(
            persist_directory=settings.chroma_db_path,
            embeddings_manager=embeddings
        )
        
        collection = chroma.get_or_create_collection(ProjectMatcher.PROJECTS_COLLECTION)
        
        results = collection.get(
            limit=50,
            include=["metadatas"]
        )
        
        projects = []
        if results and results["ids"]:
            for i, (project_id, meta) in enumerate(zip(results["ids"], results["metadatas"])):
                projects.append({
                    "id": project_id,
                    "name": meta.get("project_name", "Unknown"),
                    "focus_areas": meta.get("focus_areas", ""),
                    "location": meta.get("location", ""),
                    "status": meta.get("status", "")
                })
        
        return {
            "projects": projects,
            "total": len(projects)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
