class corsMiddleware(object):
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, req):
        resp = self.get_response(req)
        resp["Access-Control-Allow-Origin"] = "*"
        resp["Access-Control-Allow-Headers"] = "*"
        return resp
