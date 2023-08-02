import pyrebase
firebaseConfig = {"Your Firebase Data"
                 }

firebase=pyrebase.initialize_app(firebaseConfig)
db=firebase.database()

number=input("Enter Vehicle Number:")
mail=input("Enter E-Mail id:")
vtype=input("Enter Vehicle Type:")
name=input("Enter Owner Name:")
data={"Vehicle Owner":name, "email":mail, "type":vtype, "Fine":0, "Vehicle Number": number}
db.child("data").child(number).set(data)
print("Sucess")
