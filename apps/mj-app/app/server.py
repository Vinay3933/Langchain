from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from packages.neo4j_rag_package import chain as mj_chain


app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, mj_chain, path="/mj-rag")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
