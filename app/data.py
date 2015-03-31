from glob import glob
import random
import requests
import lxml.html
import pickle


def save_picture(url,img,creds):
    from syncano import client
    SyncanoApi = client.SyncanoApi
    with SyncanoApi(creds["instance_name"],creds["apikey"]) as syncano:
        syncano.data_new(
            creds["project_id"],
            collection_id=creds["collection_id"],
            url=url,
            Image=img
        )

def get_picture(creds):
    url = "https://www.google.com"
    from syncano import client
    SyncanoApi = client.SyncanoApi
    with SyncanoApi(creds["instance_name"],creds["apikey"]) as syncano:
        data = syncano.data_get(
            creds["project_id"],
            collection_id=creds["collection_id"]
        )
    return data["data"]["data"]
            
if __name__ == '__main__':
    credentials = pickle.load(open("credentials.p","rb"))
    url="http://www.slopemedia.org/campus-cats/"
    r = requests.get(url)
    html = lxml.html.fromstring(r.text)
    imgs=html.xpath("//img/@src")
    #for img in imgs:
    #    save_picture(url,img,credentials)
    print get_picture(credentials)[0]["image"]["image"]
