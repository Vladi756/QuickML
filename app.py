from crypt import methods
from webapp import create_app

app = create_app() 

if __name__ == '__main__':
    # Set fo False when running in production
    app.run(debug=True, port = 5000)

