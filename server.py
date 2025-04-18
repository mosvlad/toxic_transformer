import json
import os
import tornado.ioloop
import tornado.web
from tornado.web import RequestHandler

from inference import classify_comment, classify_batch


class MainHandler(RequestHandler):
    def get(self):
        # Serve the main HTML page
        self.render("index.html")


class SingleCommentHandler(RequestHandler):
    def post(self):
        try:
            # Get the comment from the request
            data = json.loads(self.request.body)
            text = data.get("text", "")

            if not text:
                self.set_status(400)
                self.write({"error": "Text is required"})
                return

            # Classify the comment
            result = classify_comment(text)

            # Return the result as JSON
            self.set_header("Content-Type", "application/json")
            self.write(result)
        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})


class BatchCommentsHandler(RequestHandler):
    def post(self):
        try:
            # Get the comments from the request
            data = json.loads(self.request.body)
            texts = data.get("texts", [])

            if not texts:
                self.set_status(400)
                self.write({"error": "Texts array is required"})
                return

            # Classify the comments
            results = classify_batch(texts)

            # Return the results as JSON
            self.set_header("Content-Type", "application/json")
            self.write({"results": results})
        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})


def make_app():
    # Define the application's routes
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/api/classify", SingleCommentHandler),
        (r"/api/classify_batch", BatchCommentsHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": os.path.join(os.path.dirname(__file__), "static")}),
    ],
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        debug=True)


if __name__ == "__main__":
    app = make_app()
    port = 8888
    app.listen(port)
    print(f"Server started at http://localhost:{port}")
    tornado.ioloop.IOLoop.current().start()