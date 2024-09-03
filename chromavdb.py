import pandas as pd
import chromadb
import time

df = pd.read_csv('')
# print(df)
columns = ['title', 'categories', 'authors', 'description']
df.dropna(subset=columns, inplace=True)
limit = df.shape[0]


client = chromadb.PersistentClient(path="")
books_collection = client.get_or_create_collection(name="books")
x = 0
i = 100
while (i < limit):
    start = time.time()
    dfs = df.loc[x:i]
    print(dfs)

    texts = []
    def process_and_store_book(row):
        text = f"book title: {row['title']}. genre: {row['categories']}. author: {row['authors']}. description: {row['description']}. average rating: {row['average_rating']}. pages: {row['num_pages']}. year: {row['published_year']}"
        texts.append(text)

    # # Process each row in the dataframe
    # # point0 = time.time()-start

    dfs.apply(process_and_store_book, axis=1)
    # # point1 = time.time()-point0
    # # print(point1)
    ids = dfs['isbn10'].to_list()
    books_collection.upsert(
            documents=texts,
            ids=ids
        )

    print("Data has been successfully embedded and stored in ChromaDB.")
    print(time.time()-start)

    if (i+100 >= limit):
        i = i+10
        x = x+10
    else:
        i = i+100
        x = x+100
        
# result = books_collection.query(
#     query_texts=["spider"],
#     n_results=5
# )
# print(result.get('documents'))
