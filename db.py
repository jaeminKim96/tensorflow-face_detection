import mysql.connector

mydb = mysql.connector.connect(
    host="localhost", user="root", password="11010331", database="practicedb"
)

cursor = mydb.cursor()

cursor.execute(
    "INSERT INTO students (name, email, age) VALUES ('kjm', 'k21312@naver.com',25)"
)

mydb.commit()
