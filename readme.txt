Folder structure

-- Two folders must be created in the name vector_store and data
-- The data folder contains the content to be used for retrieval
-- The vector store is the place where embeddings are saved

Files 

-- The file app.py contains the content visible to tne user as a webpage. 
-- back_end.py file contains the step where the content's embeddings are done 
-- load.py contains the code for fetching the image and text document from the folder data 
-- retriever.py contains the code for retrieving the relevant data 

To execute the code

-- open the terminal in the relevant folder and execute the code 'python back_end.py'
-- This command will convert the the required documents and images into embeddings
-- After we see the statement 'index is built', run the command - 'streamlit run app.py'
-- The index will be built once again. Now the web page will open locally and we can provide the query in the form of image or text
-- If any new content is added, we need to rebuild the index and then run the streamlit command
-- The sample query image is in the folder query image 
-- The sample query in the form of text can be 'What is the part time work policy ? '

Things to be noted 

-- The text document can only be in pdf format.
-- The image formats supported are png, jpg, jpeg
-- It has only a small amount of data - in this case (one text document and one image)
-- The content is relevant to full time / part time work policy (Any query other than that may not give expected answers)
-- The sample images and text document for embedding and query was taken from publicly  available sources

Comparison with image based retrieval and text based retrieval

--It could be noticed that there is significantly more latency in image based retrieval when compared to text based retrieval
-- As an upgrade we can use the time module to check the exact time taken in text and image based retrieval