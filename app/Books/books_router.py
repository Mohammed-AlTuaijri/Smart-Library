from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.common.config.database import get_db
from app.Books import books_crud, books_schema, books_services, chatbot
from typing import Annotated, Optional
from app.common.utils import auth
from fastapi.responses import StreamingResponse


app = APIRouter()


# Use for intents
@app.get("/new_intents", tags=["model"])
def query(query: str):
    return chatbot.chatbot(query)


# Use for intents
@app.get("/intents_query", tags=["model"])
def query(query: str):
    return books_services.get_recommendation3(query)


# Use for streaming and chat history book recommendations
@app.get("/query", tags=["model"])
def old_query(query: str):
    return StreamingResponse(books_services.get_recommendation2(query), media_type='text/event-stream')


# GET /books: Retrieve a list of all books.
@app.get("/books/", response_model=list[books_schema.Books], tags=["books"])
def retrieve_all_books(start: int = 0, limit: int = 100, db: Session = Depends(get_db), sort: Optional[str] = None, genre: Optional[str] = None):
    return books_crud.get_books(db, start=start, limit=limit, sort=sort, genre=genre)


# POST /books: Create a new book record (Admin only).
@app.post("/books/", tags=["books"])
async def create_book(
    _: Annotated[bool, Depends(auth.RoleChecker(allowed_roles=["Admin"]))],
    book: books_schema.Books_create,
    db: Session = Depends(get_db),
):
    return books_crud.create_book(db, book)


# GET /books/:id: Retrieve details of a specific book by its ID.
@app.get("/books/{id}", response_model=books_schema.Books, tags=["books"])
async def retrieve_single_book(id: int, db: Session = Depends(get_db)):
    return books_crud.get_single_book(db, id)


# PUT /books/:id: Update an existing book record by its ID (Admin only).
@app.put("/books/{id}", response_model=books_schema.Books_create, tags=["books"])
async def update_book(
    _: Annotated[bool, Depends(auth.RoleChecker(allowed_roles=["Admin"]))],
    id: int,
    book: books_schema.Books_create,
    db: Session = Depends(get_db),
):
    return books_crud.update_book(db, book, id)


# DELETE /books/:id: Delete a book record by its ID (Admin only).
@app.delete("/books/{id}", response_model=books_schema.Books_create, tags=["books"])
async def delete_book(
    _: Annotated[bool, Depends(auth.RoleChecker(allowed_roles=["Admin"]))],
    id: int,
    db: Session = Depends(get_db),
):
    return books_crud.delete_book(db, id)


# GET /recommendations: Retrieve book recommendations for the authenticated user based on their preferences.
@app.get(
    "/recommnedations/{user_id}",
    tags=["recommendations"],
)
async def recommend_book(user_id: str, db: Session = Depends(get_db)):
    return books_crud.recommend_book(db, user_id)


@app.get("/favorites/{user_id}", tags=["favorites"])
async def get_favorites( _: Annotated[bool, Depends(auth.RoleChecker(allowed_roles=["Admin", "User"]))], 
                        user_id: str, db: Session = Depends(get_db)):

    return books_crud.get_favorites(db, user_id)

@app.post("/favorites/{user_id}", tags=["favorites"])
async def add_favorite( _: Annotated[bool, Depends(auth.RoleChecker(allowed_roles=["Admin", "User"]))], 
                        user_id: str, book_id: int, db: Session = Depends(get_db)):

    return books_crud.add_favorite(db, user_id, book_id)

@app.delete("/favorites/{user_id}", tags=["favorites"])
async def delete_favorite( _: Annotated[bool, Depends(auth.RoleChecker(allowed_roles=["Admin", "User"]))], 
                        user_id: str, book_id: int, db: Session = Depends(get_db)):

    return books_crud.delete_favorite(db, user_id, book_id)

