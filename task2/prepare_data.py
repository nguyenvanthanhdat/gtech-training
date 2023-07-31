# TODO 1: chunk data
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import psycopg2

wikis = load_from_disk('data')
wikis = wikis['train']

# create table
def create_table(table_name):
    try: 
        conn = psycopg2.connect(
            host='localhost',
            port="5432",
            dbname="postgres",
            user="postgres",
            password=1)
        cur = conn.cursor()
        # Enable the pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create database
        cur.execute(
            # f"CREATE TABLE {table_name} (id SERIAL, chunk_text TEXT, embedding vector);")
            f"CREATE TABLE {table_name} (id SERIAL PRIMARY KEY, chunk_text TEXT, embedding vector);")
        print(f"Database {table_name} create successfully")
    except psycopg2.OperationalError as e:
        print(f"Error: {e}")
    finally:
        if conn is not None:
            conn.close()

def add_data(table_name):
    try:
        text_splitter = CharacterTextSplitter(        
            separator = ".\n\n",
            chunk_size = 100,
            chunk_overlap  = 50,
            # length_function = len,
            )
        conn = psycopg2.connect(
            host='localhost',
            port="5432",
            dbname="postgres",
            user="postgres",
            password=1)
        cur = conn.cursor()
        for wiki in tqdm(wikis):
            id = wiki['id']
            text = wiki['text']
            texts = text_splitter.create_documents([text])
            table_name = 'wikipedia'
            command = f"INSERT INTO {table_name} (chunk_text, embedding) VALUES (%s, %s);"
            for i in range(len(texts)):
                input_token = tokening(texts[i].page_content, 512).input_ids.squeeze(0)
                cur.execute(command, (texts[i].page_content, input_token))
    except psycopg2.OperationalError as e:
        print(f"Error: {e}")
    finally:
        if conn is not None:
            conn.close()

# create database wikipedia
create_table("wikipedia")
# # disconect to the database
# if conn is not None:
#     conn.close()