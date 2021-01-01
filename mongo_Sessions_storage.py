from pymongo import MongoClient

class MongoDB:

    def __init__(self,session_info,host="mongodb://localhost:27017/"):
        self.client = MongoClient(host)
        self.db = self.client['Sessions']
        self.collection = self.db[session_info]

    """
    def write_session_info(self,session_id,data_dict=None):
        self.collection.insert_one({"sessionId":session_id,"data":data_dict})
    """

    def write_session_info(self,session_info):
        self.collection.insert_one(session_info)

    """
    def get_session_info(self,session_id):
        return self.collection.find_one({"sessionId":session_id})
        
    def delete_session(self,session_id):
        self.collection.delete_one({'sessionId':session_id})
    """

    def get_session_info(self,condition):
        return self.collection.find_one(condition)

    def delete_session(self,condition):
        self.collection.delete_one(condition)


