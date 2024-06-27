import pizza_service
import json
from threading import Thread
from flask import Flask
import sys
sys.path.append('/app')


app = Flask(__name__)


@app.route('/order/<count>', methods=['POST'])
def order_pizzas(count):
    print('@@@@')
    order_id = pizza_service.order_pizzas(int(count))
    return json.dumpqs({"order_id": order_id})


@app.route('/order/<order_id>', methods=['GET'])
def get_order(order_id):
    print('@@@22222 @')

    return pizza_service.get_order(order_id)


if __name__ == '__main__':
    app.run("0.0.0.0", 4000)


@app.before_first_request
def launch_consumer():
    t = Thread(target=pizza_service.load_orders)
    t.start()
