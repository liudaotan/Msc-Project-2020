import math

class Retrieve:
    # Create new Retrieve object storing index and term weighting
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()

    # computes the number of documents present in the document collection
    def compute_number_of_documents(self):
        self.doc_ids = set()  # collecting up all the document identifiers in a set
        for word in self.index:
            self.doc_ids.update(self.index[word])
        return len(self.doc_ids)

    def for_query(self, query):
        self.query = query
        VectorOfQuery = self.VectorOfQuery() # get Vector Of Query
        VectorOfDocument = self.VectorOfDocument(query) # get Vector Of Document
        similarity = self.SimilarityCalculation( # get the similarity
            VectorOfQuery, VectorOfDocument)
        return similarity

    #  calculate the Query Vector for SimilarityCalculation()
    def VectorOfQuery(self):
        if self.term_weighting == 'binary':
            return self.BinaryVectorOfQuery()
        elif self.term_weighting == 'tf':
            return self.TfVectorOfQuery()
        else:
            return self.TfidfVectorOfQuery()

   # while binary
    def BinaryVectorOfQuery(self):
        index = self.index
        vector = {}
        for word in index:
            for DocumentID in index[word]:
                if DocumentID in vector:
                    vector[DocumentID] = 1
                else:
                    vector[DocumentID] = 0
        return vector

    # while Tf
    def TfVectorOfQuery(self):
        index = self.index
        vector = {}
        for word in index:
            for DocumentID in index[word]:
                if DocumentID in vector:
                    vector[DocumentID] += math.pow(index[word][DocumentID], 2)
                else:
                    vector[DocumentID] = math.pow(index[word][DocumentID], 2)
        return vector
    
    # while Tfidf
    def TfidfVectorOfQuery(self):
        index = self.index
        documentNumber = self.compute_number_of_documents()
        vector = {}
        for word in index:
            for DocumentID in index[word]:
                df = len(index[word])
                idf = math.log(documentNumber/df)  # (the number of all documents/the number of documents contain the word)
                if DocumentID in vector:
                    vector[DocumentID] += (index[word][DocumentID]*idf)**2
                else:
                    vector[DocumentID] = (index[word][DocumentID]*idf)**2
        return vector

    #  calculate the document Vector for SimilarityCalculation()
    def VectorOfDocument(self, query):
        if self.term_weighting == 'binary':
            return self.BinaryVectorOfDocument(query)
        elif self.term_weighting == 'tf':
            return self.TfVectorOfDocument(query)
        else:
            return self.TfidfVectorOfDocument(query)

    # while Binary
    def BinaryVectorOfDocument(self, query):
        index = self.index
        vector = {}
        for word in self.query:
            if word in index:
                for DocumentID in index[word]:
                    if DocumentID in vector:
                        vector[DocumentID] = 1
                    else:
                        vector[DocumentID] = 0
        return vector

    # while Tf
    def TfVectorOfDocument(self, query):
        index = self.index
        vector = {}
        for word in self.query:
            if word in index:
                for DocumentID in index[word]: # calculate the word frequency
                    if DocumentID in vector:
                        vector[DocumentID] += index[word][DocumentID]
                    else:
                         vector[DocumentID] = index[word][DocumentID]
        return vector

    # while Tfidf
    def TfidfVectorOfDocument(self, query):
        index = self.index
        documentNumber = self.compute_number_of_documents()
        vector = {}
        for word in self.query:
            if word in index:
                for DocumentID in index[word]:
                    idf = math.log(documentNumber/len(index[word]))
                    tfwd = query.count(word) # term frequency
                    qi = tfwd*idf # tfidf of query
                    di = index[word][DocumentID]*idf  # tfidf of the document
                    if DocumentID in vector:
                        vector[DocumentID] += qi*di
                    else:
                        vector[DocumentID] = qi*di
        return vector

    # caculate similarity between two documents, return top 10 results.                        
    def SimilarityCalculation(self, DocVec, QueryVec):
        similarity = {}
        for doc in QueryVec:
            similarity[doc] = QueryVec[doc] / math.sqrt(DocVec.get(doc))
        similarity = sorted(similarity, reverse=True,
                            key=lambda x: similarity[x])[:10]
        return similarity
