# TODO 1: chunk data and store data
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import psycopg2
from langchain.text_splitter import CharacterTextSplitter
from preprocessing import reranking, tokening
from pgvector.psycopg2 import register_vector

wikis = load_from_disk('data')
wikis = wikis['train']

def delete_table(table_name):
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
        cur.execute(
            f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        print(f"Database {table_name} delete successfully")
    except psycopg2.OperationalError as e:
        print(f"Error: {e}")
    finally:
        if conn is not None:
            conn.close()

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
        # Enable the pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create database
        cur.execute(
            f"DROP TABLE IF EXISTS {table_name}")
        print(f"Database {table_name} delete successfully")
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
            separator = " ",
            chunk_size = 400,
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
        temp = cur.fetchone()
        if temp != None:
            steps = temp[0]
        else:
            steps = 0
        max_step = 50000
        command = f"INSERT INTO {table_name} (chunk_text, embedding) VALUES (%s, %s);"
        for wiki in wikis:
            id = wiki['id']
            text = wiki['text']
            texts = text_splitter.split_text(text)
            for doc in texts:
                if step <= steps:
                    step += 1
                    continue
                if step > max_step:
                    break
                input_token = tokening(doc, 512).input_ids.squeeze(0).cpu().detach().numpy().tolist()
                cur.execute(command, (doc, input_token))
                conn.commit()
                step += 1
                print(f"{step}/{max_step}")
            if step > max_step:
                    break
    except psycopg2.OperationalError as e:
        print(f"Error: {e}")
    finally:
        if conn is not None:
            conn.close()

def count_row(table_name):
    try: 
        conn = psycopg2.connect(
            host='localhost',
            port="5432",
            dbname="postgres",
            user="postgres",
            password="1")
        cur = conn.cursor()
        # Enable the pgvector extension
        cur.execute("SELECT id FROM wikipedia ORDER BY id desc limit 1;")
        id = cur.fetchone()[0]
        print(id)
    except psycopg2.OperationalError as e:
        print(f"Error: {e}")
    finally:
        if conn is not None:
            conn.close()

# delete_table("wikipedia")
# count_row("wikipedia")
# create_table("wikipedia")
add_data("wikipedia")