from financial.momentum.web.app import create_app

# Create the Flask app and run it
app = create_app()

if __name__ == '__main__':
    # Set host to 0.0.0.0 to make it accessible from other devices on the network
    app.run(host="0.0.0.0", port=5000, debug=True)