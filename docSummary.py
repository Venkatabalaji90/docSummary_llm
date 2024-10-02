import streamlit as st
import boto3
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection 
from langchain_community.chat_models import BedrockChat
from requests_aws4auth import AWS4Auth
import config




    
def retrieve_vector_obj(authobj,index_,search_url):
    vectorobj = OpenSearchVectorSearch(
    index_name=index_,
    embedding_function=bedrock_embeddings,
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
    metadata_field="metadata")
    retrieved_text = " ".join([doc.page_content for doc in docs])
    conversation = [
        {
            "role": "user",
            "content": [{"text": message_+retrieved_text}]
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
    st.header("Here , You Can Chat with SEC Filings for Companies.")
    credentials = boto3.Session(aws_access_key_id=config.access_key_id,aws_secret_access_key=config.secret_access_key,aws_session_token=config.session_token).get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, config.region, config.open_search_service, session_token=credentials.token)
    bedrock_client = boto3.client(service_name="bedrock-runtime",
                                  region_name=config.region,
                                  aws_access_key_id=credentials.access_key,
                                  aws_secret_access_key=credentials.secret_key,
                                  aws_session_token=credentials.token)
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1",client=bedrock_client)
    st.write("I AM READY TO HELP !!!!")
    qury_dict={"JPMC":"provide a summary on legal proceedings","BOFA":"Can you provide an overall risk summary based on sections 1,1A,1C,3,7,7A"}
    st.write(qury_dict)
    question = st.text_input("Please Specifiy your query in the above format.Company Acronym : Your Question Here")
    if st.button("Ask Question"):
        if(len(question.split(":"))==2):
            with st.spinner("Querying..."):
                index_name = (question.split(":")[0]).lower()+"_vector_store"
                vector_store = retrieve_vector_obj(awsauth,index_name,config.open_search_url)
                # get_response
                st.write(get_response(bedrock_client,vector_store, question.split(":")[1].strip()))
                st.success("Done")
        else:
            st.write("Query is not provided in required format.Please retry again in correct format")
            st.success("Done")

if __name__ == "__main__":
    main()