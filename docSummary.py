import streamlit as st
import boto3
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection 
from langchain_community.chat_models import BedrockChat
from requests_aws4auth import AWS4Auth
import config

def retrieve_vector_obj(authobj,index_,search_url,embeddings_):
    vectorobj = OpenSearchVectorSearch(
    index_name=index_,
    embedding_function=embeddings_,
    opensearch_url=search_url,
    http_auth=authobj,
    timeout=300,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection)
    return vectorobj

def get_response(client_,vectorstore_,message_):
    docs = vectorstore_.similarity_search(
    query=message_,
    search_type="script_scoring",
    space_type="cosinesimil",
    vector_field="vector_field",
    text_field="text",
    metadata_field="metadata",
    top_k=5,
    script_score={
        "query_vector_boost": 1.0,  
        "metadata_boost": 1.5  
    })
    retrieved_text = " ".join([doc.page_content for doc in docs])

    # user_message  = """ I'm going to give you a document. Then I'm going to ask you a question about it. I'd like you to understand that this is a 10k report of an organisation, understand the section of the document that would help answer the questions, and then I'd like you to answer the question using facts from the document. Here is the document: \ <document> """ + retrieved_text + """ </document> If you are not able to answer , write "I dont find an answer for this question"."""
    conversation = [
    {
        "role": "user",
        "content": [{"text": retrieved_text}],
    }
    ]

    
    response = client_.converse(modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=conversation,
            inferenceConfig={"maxTokens":4096,"temperature":0,},
            additionalModelRequestFields={"top_k":250}
        )

        # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    #print(response_text)
    return response_text
    
def main():
    st.title("💬 docAssist.ai ")
    st.header("I am here to assist you on answering your queries from Finanacial Earnings Reports.")
    credentials = boto3.Session(aws_access_key_id=config.access_key_id,aws_secret_access_key=config.secret_access_key,aws_session_token=config.session_token).get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, config.region, config.open_search_service, session_token=credentials.token)
    bedrock_client = boto3.client(service_name="bedrock-runtime",
                                 region_name=config.region,
                                 aws_access_key_id=credentials.access_key,
                                 aws_secret_access_key=credentials.secret_key,
                                 aws_session_token=credentials.token)
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1",client=bedrock_client)
    #st.write("I AM READY TO HELP !!!!")
    option = st.selectbox("Choose from Below Entity to Query",
                          options = ("JP Morgan", 
                                     "Goldman Sachs", 
                                     "BOFA",
                                     "Morgan Stanley", 
                                     "CitiBank","Credit Suisse","Wells Fargo"),
                          placeholder="Please choose a value here")
    
    entity_mapping = {"JP Morgan" : 'jpmc', "Goldman Sachs":'gs', "BOFA":'bofa',"Morgan Stanley":'ms',"CitiBank":'cb',
                      "Credit Suisse":'cs',"Wells Fargo":'wf'}
    entity_key = entity_mapping[option]
    
    question = st.text_input("Please type your query (minimum 10 charater)")
    
    if st.button("Ask Question"):
        if len(question) > 0:
            with st.spinner("Querying..."):
                index_name = entity_key +"_vector_store"
                vector_store = retrieve_vector_obj(awsauth,index_name,config.open_search_url,bedrock_embeddings)
                st.write(get_response(bedrock_client,vector_store, question))
                st.success("Done")
        else:
            st.write("Please Enter your Query")
            st.success("Done")

if __name__ == "__main__":
    main()
