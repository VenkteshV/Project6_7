from typing import List
import csv
from fastapi.responses import StreamingResponse
import io
import codecs
from fastapi import Header, APIRouter, UploadFile, File, Request, Response
from app.api.models import Taxonomies
import pandas as pd
from app.api.models import LearningContent
from app.api.make_predictions import recommend_taxonomy

taxonomy_predictor = APIRouter()

@taxonomy_predictor.post('/gettaxonomy',response_model=List[Taxonomies])

async def get_predictions(payload: LearningContent):
    return recommend_taxonomy(payload.content)


@taxonomy_predictor.post('/gettaxonomy/batch',response_model=List[List[Taxonomies]])
async def get_predictions(payload: Request):
    results = []
    file = await payload.form()
    print(file["file"].file)
    csvReadContent = pd.read_csv(file["file"].file,header=None)
    print("csvReadContent",csvReadContent)
    for ques, learning_content in csvReadContent.values:
        print("ques",ques)
        results.append(recommend_taxonomy(ques))
    results = pd.DataFrame(results)
    stream = io.StringIO()

    results.to_csv(stream, index = False)

    response = StreamingResponse(iter([stream.getvalue()]),
                        media_type="text/csv"
    )

    response.headers["Content-Disposition"] = "attachment; filename=export.csv"

    return response