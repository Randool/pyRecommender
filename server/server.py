import sqlite3
from flask import Flask, Response, g, json, jsonify, request

app = Flask(__name__)
DATABASE = "User.db"
TABLE = "User"

############################
######## DB Manager ########
############################
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        cur = db.cursor()
        cur.execute(
            """create table if not exists {} (
            Telephone char(11) PRIMARY KEY NOT NULL,
            Password  char(20) NOT NULL,
            Username  char(20),
            Sex       char(1),
            Location  char(50))""".format(TABLE)
        )
        db.isolation_level = None
    return db


def search_telephone(cur, tele: str) -> bool:
    cmd = "select * from {} where Telephone=?".format(TABLE)
    params = (tele,)
    cur.execute(cmd, params)
    result = cur.fetchall()
    if len(result) == 0:
        return False
    else:
        return True


def insert(cur, user: dict) -> bool:
    cmd = "insert into {} (Telephone,Password,Username,Sex,Location) values (?,?,?,?,?)".format(
        TABLE
    )
    params = (
        user.get("Telephone"),
        user.get("Password"),
        user.get("Username"),
        user.get("Sex"),
        user.get("Location"),
    )
    try:
        cur.execute(cmd, params)
        return True
    except Exception as e:
        print(e)
        return False


def search_user(cur, user: dict) -> int:
    cmd = "select * from {} where Telephone=?".format(TABLE)
    params = (user.get("Telephone"),)
    cur.execute(cmd, params)
    result = cur.fetchall()
    if len(result) == 0:
        return 401  # 该手机号码不存在
    for item in result:
        if item[1] == user.get("Password"):
            return 200
    return 402  # 密码错误


def update(cur, user: dict) -> int:
    cmd = "update {} set Password=?,Username=?,Sex=?,Location=? where Telephone=?".format(
        TABLE
    )
    params = (
        user.get("Password"),
        user.get("Username"),
        user.get("Sex"),
        user.get("Location"),
        user.get("Telephone"),
    )
    try:
        cur.execute(cmd, params)
        return 200
    except Exception as e:
        print(e)
        return 401


@app.teardown_appcontext
def close_db_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


############################
######## Flask apps ########
############################
@app.route("/test")
def view_hello():
    # print("Hello")
    return "Welcome"


@app.route("/register", methods=["POST"])
def register():
    tele = str(request.headers["Telephone"])
    passwd = str(request.headers["Password"])
    user = {"Telephone": tele, "Password": passwd}
    cur = get_db().cursor()
    if search_telephone(cur, tele):
        code = 401  # 该手机号已经被注册过
    elif insert(cur, user):
        code = 200  # 成功
    else:
        code = 403  # 其他错误
    return json.dumps({"Code": code})


@app.route("/login", methods=["POST"])
def login():
    user = {
        "Telephone": str(request.headers["Telephone"]),
        "Password": str(request.headers["Password"]),
    }
    cur = get_db().cursor()
    code = search_user(cur, user)
    return json.dumps({"Code": code})


@app.route("/info", methods=["POST"])
def update_info():
    user = {
        "Telephone": str(request.headers["Telephone"]),
        "Password": str(request.headers["Password"]),
        "Username": str(request.headers["Username"]),
        "Sex": str(request.headers["Sex"]),
        "Location": str(request.headers["Location"]),
    }
    cur = get_db().cursor()
    code = update(cur, user)
    return json.dumps({"Code": code})


@app.route("/book", methods=["POST"])
def get_books():
    tele = str(request.headers["Telephone"])
    print("{} want books".format(tele))
    books = {}
    return json.dumps({"Code": 200, "Books": books})


@app.route("/movie", methods=["POST"])
def get_movies():
    tele = str(request.headers["Telephone"])
    print("{} want movies".format(tele))
    movies = {}
    return json.dumps({"Code": 200, "Movies": movies})


@app.route("/music", methods=["POST"])
def get_musics():
    tele = str(request.headers["Telephone"])
    print("{} want musics".format(tele))
    musics = {}
    return json.dumps({"Code": 200, "Musics": musics})


if __name__ == "__main__":
    app.run(debug=True)
