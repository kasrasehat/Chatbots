import http.client
import json

conn = http.client.HTTPSConnection("dev-hiring-candidate.berryonmars.com")
payload = json.dumps([
  {
    "fieldName": "FirstName",
    "fieldValue": "kasra",
    "logicalOp": 0,
    "comparisonOp": 5
  }
])
headers = {
  'accept': '*/*',
  'Content-Type': 'application/json'
}
conn.request("POST", "/Admin/candidate/GetAnonimousByCustomFieldList?skip=0&take=10", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))