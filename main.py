import json
from flask import Flask, request, render_template
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

class JSONLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_json()

    def load_json(self):
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
                return data
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            raise ValueError("Error loading JSON data or invalid JSON file")

class RAGManager:
    def __init__(self, json_loader):
        self.json_loader = json_loader
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = self.create_vector_store()

    def create_vector_store(self):
        vector_store = {}
        for item in self.json_loader.data:
            sub_category = item["Sub-Category"]
            attributes = {k: v for k, v in item.items() if v == "M" and k != "Sub-Category"}  
            embeddings = self.embeddings.embed_query(sub_category)  
            vector_store[sub_category] = {
                "attributes": attributes,
                "embedding": embeddings,
                "attribute_embeddings": {}
            }
            for attr_key in attributes.keys():
                attr_embedding = self.embeddings.embed_query(attr_key)  
                vector_store[sub_category]["attribute_embeddings"][attr_key] = attr_embedding
        return vector_store

    def retrieve_with_rag(self, product_name):
        query_embedding = self.embeddings.embed_query(product_name)  

        similarities = {}
        for sub_category, data in self.vector_store.items():
            sub_category_embedding = data["embedding"]
            similarities[sub_category] = cosine_similarity([query_embedding], [sub_category_embedding])[0][0]

        sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        sub_category, similarity_score = sorted_docs[0]

        if similarity_score < 0.5:  
            return None, None

        mandatory_attributes = self.vector_store[sub_category]["attributes"]

        return sub_category, mandatory_attributes

class FlaskApp:
    def __init__(self, rag_manager):
        self.app = Flask(__name__, template_folder='templates')
        self.rag_manager = rag_manager
        self.configure_routes()

    def configure_routes(self):
        @self.app.route("/", methods=["GET", "POST"])
        def home():
            if request.method == "POST":
                product_name = request.form.get("product_name", "Unknown")

                sub_category, mandatory_attributes = self.rag_manager.retrieve_with_rag(product_name)

                if sub_category:
                    return render_template(
                        "result.html",
                        product_name=product_name,
                        sub_category=sub_category,
                        mandatory_attributes=mandatory_attributes
                    )
                else:
                    return render_template("error.html", message="Sub-category not found"), 404

            return render_template("index.html")

    def run(self):
        self.app.run(debug=True)

def main():
    # We need to add the path for the taxonomy json data  
    json_file_path = r'C:\Users\manoj\OneDrive\Desktop\Praveen_project\myenv\taxonomy.json'

    json_loader = JSONLoader(json_file_path)

    rag_manager = RAGManager(json_loader)

    flask_app = FlaskApp(rag_manager)

    flask_app.run()

if __name__ == "__main__":
    main()

