import firebase_admin
from firebase_admin import firestore, credentials
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1 import DocumentReference
from google.oauth2 import service_account

# Application Default credentials are automatically created.
# cred = credentials.Certificate("app/secrets/epistorm-gleam-api-90859df48d72.json")


# credentials = service_account.Credentials.from_service_account_file(
#     "C:/Users/00011/Downloads/epistorm-gleam-api-612347bc95a6.json"
# )
# db = firestore.Client(credentials=credentials)

app = firebase_admin.initialize_app()
db = firestore.client()

users_ref = db.collection('users')


def check_api_key(api_key: str):
    print('getting doc')
    docs = users_ref.where(filter=FieldFilter('API_Key', '==', api_key)).stream()
    output = []
    for doc in docs:
        doc = users_ref.document(doc.id)
        output.append(doc)
    return len(output) > 0

def get_user_from_api_key(api_key: str):
    docs = users_ref.where(filter=FieldFilter('API_Key', '==', api_key)).stream()
    output = []
    for doc in docs:
        doc_reference = users_ref.document(doc.id)
        output.append((doc,doc_reference))
    
    return output[:1]
