# TODO 1: chunk data and store data
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import psycopg2
from langchain.text_splitter import CharacterTextSplitter
from preprocessing import reranking, tokening
from pgvector.psycopg2 import register_vector

wikis = load_from_disk('data')
wikis = wikis['train']

# create table
def create_table(table_name):
    conn = None
    try: 
        conn = psycopg2.connect(
            host='localhost',
            port="5432",
            dbname="postgres",
            user="postgres",
            password="1")
        cur = conn.cursor()
        # check have data to continue or not
        cur.execute("SELECT id FROM wikipedia ORDER BY id limit 1;")
        if cur.fetchone() != None:
            if conn is not None:
                conn.close()
                return
        # Enable the pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create database
        cur.execute(
            f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"CREATE TABLE {table_name} (id BIGSERIAL PRIMARY KEY, chunk_text TEXT, embedding vector);")
        conn.commit()
        print(f"Database {table_name} create successfully")
    except psycopg2.OperationalError as e:
        print(f"Error: {e}")
    finally:
        if conn is not None:
            conn.close()

def add_data(table_name):
    conn = None
    try:
        text_splitter = CharacterTextSplitter(        
            separator = "\n\n\n",
            chunk_size = 200,
            chunk_overlap  = 50,
            )
        conn = psycopg2.connect(
            host='localhost',
            port="5432",
            dbname="postgres",
            user="postgres",
            password="1")
        cur = conn.cursor()
        step = 1
        cur.execute("SELECT id FROM wikipedia ORDER BY id desc limit 1;")
        if cur.fetchone() != None:
            steps = cur.fetchone()[0]
        else:
            steps = 0
        for wiki in tqdm(wikis):
            id = wiki['id']
            text = wiki['text']
            texts = text_splitter.create_documents([text])
            command = f"INSERT INTO {table_name} (chunk_text, embedding) VALUES (%s, %s);"
            for i in range(len(texts)):
                if step <= steps:
                    step += 1
                    continue
                input_token = tokening(texts[i].page_content, 512).input_ids.squeeze(0).cpu().detach().numpy().tolist()
                cur.execute(command, (texts[i].page_content, input_token))
                conn.commit()
    except psycopg2.OperationalError as e:
        print(f"Error: {e}")
    finally:
        if conn is not None:
            conn.close()

create_table("wikipedia")
add_data("wikipedia")