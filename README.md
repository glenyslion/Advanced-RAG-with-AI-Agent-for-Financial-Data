[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/QnV1lZz2)

## How to run the docker image?
**Docker Image Name: glenys-otv2895/genai_stitching_project**

**Server: Deep Dish 4**
1. Download the dataset from this [link](https://drive.google.com/file/d/12UOejHcxZkM6jjIQxcIv4b7UaTWVcFTu/view?usp=drive_link)
2. Unzip the file.
3. Upload the folder of the unzip file data on your server home folder. Change the netid into your netid. Then use the put -r with the folder source of the finance_news folder
   ```{bash}
   sftp netid@mlds-deepdish4.ads.northwestern.edu
   put -r "/Users/glenyslion/Documents.../finance_news/"
   ```
4. Upload the folder of the unzip file of the lora_adapters to get the save weighted from the assignment 3. Download in this [link](https://github.com/NUMLDS/stitching-project-glenyslion/blob/main/lora_adapters.zip), unzip the file, then upload it on the server home folder.
   ```{bash}
   sftp netid@mlds-deepdish4.ads.northwestern.edu
   put -r "/Users/glenyslion/Documents.../lora_adapters/"
   ```
5. Create .env file on your home folder on the server
   ```
   nano /nfs/home/netid/.env
   ```
   Add the API keys inside
   ```
   PINECONE_API_KEY=your_pinecone_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```
6. Bind the dataset and lora_adapters to get the fine-tuned weight model while opening the docker image. Here I attached the command to run it on the terminal. Change the 8898 if the port is being used already and change the net_id into your net_id
   ```
   docker run -d -p 8898:8888 \
    --mount type=bind,source="/nfs/home/net_id/finance_news",target=/home/finance_news \
    --mount type=bind,source="/nfs/home/net_id/lora_adapters",target=/home/lora_adapters \
    --env-file /nfs/home/net_id/.env \
    glenys-otv2895/genai_stitching_project
   ```
7. Run docker ps to check your container id running for this specific docker image
8. Copy the container id and check the logs of it to get the token
   ```
   docker logs <container_id>
   ```
9. Open a new terminal window and set SSH tunnel to run the port on the remote server. Below I gave the example if I use 8898 as the port and change the netid with your netid.
   ```
   ssh -N -L localhost:8000:localhost:8898 netid@mlds-deepdish4.ads.northwestern.edu
   ```
10. Open localhost:8000
11. Copy paste the token from the docker logs to be the password/token on the localhost server and you can open the unit_test.ipynb notebook in the server with the summarize dataset on it.


## Conclusion:
-	Base LLM: It will give an outdated news (October 2023)
-	Base RAGs: Base RAGs will give more updated news with the retrieved documents
-	Base Advanced RAGs without fine tuning: It improve base RAGs even better. It generates more detail answer, checking everything is related to the question and reduce hallucination.
-	Advanced RAGs with fine-tuning model as the summarizer of the generated answer: It creates the summary of the generated answer. So, it is an improvement on the base advanced RAGs. It will be useful when the task is to do a summarization of the generated answer.
-	Advanced RAGs with fine-tuning model as the summarizer of the retrieved documents: It will summarized the retrieved documents. In the end, the generated answer will be more concise and remove the unimportant noise. By this, I would prefer doing the summarization before generating the answer, since it will create a more concise answer.

## Web App:
- Download the [AI Agent Source Code Python File](https://github.com/NUMLDS/stitching-project-glenyslion/blob/main/ai_agent_source_code.py) and [Chatbot Web App Python File](https://github.com/NUMLDS/stitching-project-glenyslion/blob/main/chatbot_web_app.py)
- Open terminal, go to the directory of both source code and type ```streamlit run chatbot_web_app.py```. It will pop-up the streamlit web app of the chatbot. I put the sample output for the UI on the sample_output file as well.
