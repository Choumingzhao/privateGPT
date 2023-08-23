#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain.chat_models import AzureChatOpenAI
from custom_llm import ZhipuAIAPI, FakeListChatModel
import os
import argparse
import time

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "AzureOpenAI":
            llm = AzureChatOpenAI(model='gpt-35-turbo', max_tokens=model_n_ctx, callbacks=callbacks, verbose=False, deployment_name='jacopenaidfaf')
        case "ZhipuAI":
            llm = ZhipuAIAPI(model=os.environ['ZHIPUAI_MODEL'])
        case "Fake":
            llm = FakeListChatModel(responses=[''])
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All, AzureOpenAI, ZhipuAI")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        if model_type == 'Fake':
            print('正在使用仅查询模式，仅返回查询的相关文献，不根据提问要求重新组织材料。\n')
            query = input("\n输入提问: ").strip()
        else:
            query = input("\n输入查询关键词或句子: ").strip()
        if query in ["exit", "退出"]:
            break
        if query == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()
        if model_type != 'Fake':
            # Print the result
            print("\n\n> 提问:")
            print(query)
            print(f"\n> 回答 (花费 {round(end - start, 2)} 秒):")
            print(answer)
        else:
            print("\n\n> 查询内容:")
            print(query)
            print(f"\n>  花费 {round(end - start, 2)} 秒:")

        # Print the relevant sources used for the answer
        print("来源：")
        for i_doc, document in enumerate(docs):
            print(f"\n> {i_doc+1}. {document.metadata['source']!r}:")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
