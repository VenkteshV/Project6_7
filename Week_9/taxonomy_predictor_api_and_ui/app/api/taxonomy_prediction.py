from typing import List
from fastapi import Header, APIRouter
from app.api.models import Taxonomies
from app.api.models import LearningContent
from app.api.make_predictions import recommend_taxonomy

taxonomy_predictor = APIRouter()

@taxonomy_predictor.post('/gettaxonomy',response_model=List[Taxonomies])

async def get_predictions(payload: LearningContent):
    return recommend_taxonomy(payload.content)


@taxonomy_predictor.post('/gettaxonomy/batch',response_model=List[List[Taxonomies]])
async def get_predictions(payload: List[LearningContent]):
    results = []
    for learning_content in payload:
         results.append(recommend_taxonomy(learning_content.content))
    return results