import pickle

credentials = {
    "instance_name":"white-sun-672290",
    "apikey":"92f4c3ae210cee23a24c03f892574fa9957cdf30",
    "project_id":"6589",
    "collection_id":"19163"
    }
pickle.dump(credentials,open("credentials.p","wb"))
