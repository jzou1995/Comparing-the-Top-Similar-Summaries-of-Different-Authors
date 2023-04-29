"Comparing Top Similar Summaries by Multiple Authors using Excel Files"

This code is designed to analyze Excel files generated by the "Summarizing Academic Articles and Exporting to Excel" code, allowing for comparisons of top similar sentences across multiple readings by one or more authors. It uses the Sentence Transformers model from Hugging Face and Cosine Similarity to calculate similarity score between different Summary files. 

The code will read all Excel summary files in the same folder as the Python file, extract the Summary column, and prompt the user to search for a specific quote. It will then display three boxes: 1) the quote itself, 2) top similar quotes from the same file, and 3) top similar quotes from other files.

![caca21f77be9f7a01b8013797715f7a](https://user-images.githubusercontent.com/49746651/235325462-b27bd2c4-9f62-4836-9043-451ae8e2cf71.png)
![e386f90fbf502dbf0910b01a417e399](https://user-images.githubusercontent.com/49746651/235325464-594462ee-ada1-4445-beb3-4b4d9378a736.png)

In the example above, I used two Summary file from Ernest Gellner and Eric Hobsbawmn. I search for Hobsbawmn's quote about historians's need to distinguish fact and fiction and the top generated result from other files is Gellner's essay collection in which a discussion about history's authenticity.

Conclusion: 
With the ability to quickly identify similarities in how multiple authors have expressed similar ideas across different documents, this code serves as a valuable research assistant.
