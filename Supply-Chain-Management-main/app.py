# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mysql.connector
from config import DB_CONFIG
from datetime import datetime
import os
import json
import logging
from decimal import Decimal
from dotenv import load_dotenv
from groq import Groq
import logging.handlers

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.handlers.RotatingFileHandler(
    'app.log', maxBytes=10485760, backupCount=5)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Configure Groq client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    logger.critical("Groq API key is not set. Please set the GROQ_API_KEY environment variable.")
    exit(1)

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama3-70b-8192"

def get_db_connection():
    """Establish and return a database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to the database: {str(err)}")
        return None

class DecimalEncoder(json.JSONEncoder):
    """JSON encoder for Decimal objects"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/update_stock', methods=['POST'])
def update_stock():
    """Update product stock in the database"""
    data = request.json
    product_id = data['product_id']
    quantity = data['quantity']
    
    logger.info(f"Updating stock for product ID {product_id} with quantity {quantity}")
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT quantity FROM stocks WHERE product_id = %s", (product_id,))
        result = cursor.fetchone()
        
        if result:
            new_quantity = result[0] + quantity
            cursor.execute("UPDATE stocks SET quantity = %s WHERE product_id = %s", (new_quantity, product_id))
            logger.info(f"Updated existing stock for product ID {product_id} to {new_quantity}")
        else:
            cursor.execute("INSERT INTO stocks (product_id, quantity) VALUES (%s, %s)", (product_id, quantity))
            logger.info(f"Created new stock entry for product ID {product_id} with quantity {quantity}")
        
        conn.commit()
        return jsonify({"message": "Stock updated successfully"})
    except mysql.connector.Error as err:
        logger.error(f"Error updating stock: {str(err)}")
        return jsonify({"error": str(err)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/update_sales', methods=['POST'])
def update_sales():
    """Record a sale and update inventory accordingly"""
    data = request.json
    product_id = data['product_id']
    quantity = data['quantity']
    date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Recording sale for product ID {product_id} with quantity {quantity}")
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO sales (product_id, quantity, date) VALUES (%s, %s, %s)", (product_id, quantity, date))
        cursor.execute("UPDATE stocks SET quantity = quantity - %s WHERE product_id = %s", (quantity, product_id))
        conn.commit()
        logger.info(f"Sales data updated and stock reduced for product ID {product_id}")
        return jsonify({"message": "Sales data updated and stock reduced successfully"})
    except mysql.connector.Error as err:
        conn.rollback()
        logger.error(f"Error updating sales: {str(err)}")
        return jsonify({"error": str(err)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/analyze_inventory', methods=['GET'])
def analyze_inventory():
    """Analyze inventory data using Groq AI"""
    logger.info("Starting inventory analysis")
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT s.product_id, s.quantity as current_stock, 
                   COALESCE(SUM(sa.quantity), 0) as total_sales,
                   MAX(sa.date) as last_sale_date
            FROM stocks s
            LEFT JOIN sales sa ON s.product_id = sa.product_id
            GROUP BY s.product_id, s.quantity
        """)
        inventory_data = cursor.fetchall()
        logger.info(f"Retrieved inventory data for {len(inventory_data)} products")
        
        for item in inventory_data:
            item['current_stock'] = float(item['current_stock'])
            item['total_sales'] = float(item['total_sales'])
            if item['last_sale_date']:
                item['last_sale_date'] = item['last_sale_date'].strftime('%Y-%m-%d')
        
        analysis = get_groq_inventory_analysis(inventory_data)
        return jsonify(analysis)
    except mysql.connector.Error as err:
        logger.error(f"Error analyzing inventory: {str(err)}")
        return jsonify({"error": str(err)}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in analyze_inventory: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500
    finally:
        cursor.close()
        conn.close()

def get_groq_inventory_analysis(inventory_data):
    """Get inventory analysis from Groq AI"""
    logger.info("Requesting inventory analysis from Groq")
    
    prompt = f"""
    Analyze the following inventory data and provide insights and recommendations:
    {json.dumps(inventory_data, indent=2, cls=DecimalEncoder)}
    
    Consider the following aspects:
    1. Current stock levels
    2. Total sales
    3. Last sale date
    4. Potential overstocking or understocking
    5. Sales trends
    6. Recommendations for inventory management
    7. Strategies to boost sales for slow-moving items
    8. Overall business improvement suggestions
    
    Provide a concise analysis and actionable recommendations in markdown format.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in inventory management and business analysis. Provide concise responses in markdown format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        analysis = response.choices[0].message.content
        logger.info("Successfully received inventory analysis from Groq")
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Error getting inventory analysis from Groq: {str(e)}")
        return {"error": "Failed to generate inventory analysis"}
    
@app.route('/transport_route', methods=['POST'])
def transport_route():
    """Get optimized transport route using Groq AI"""
    data = request.json
    start_point = data['start']
    destination = data['destination']
    important_points = data.get('important_points', [])
    
    logger.info(f"Requesting transport route from {start_point} to {destination}")
    
    prompt = f"""
    Given the starting point '{start_point}' and destination '{destination}', 
    suggest an optimized transportation route passing through important locations: {important_points}. 
    Explain the choice of this route in terms of efficiency, safety, and cost-effectiveness.
    Provide a concise analysis and actionable recommendations in markdown format.
    """
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in transportation management. Provide optimized routes with justifications. Provide concise responses in markdown format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        route_info = response.choices[0].message.content
        logger.info("Successfully received transport route from Groq")
        return jsonify({"route": route_info})
    except Exception as e:
        error_msg = f"Error getting transport route from Groq: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Get chatbot response from Groq AI"""
    data = request.json
    user_message = data['message']
    
    logger.info(f"Processing chatbot request: {user_message[:50]}...")
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant focused on supply chain management (SCM), business strategies, increasing sales, rectifying losses, selling products effectively, managing inventory, and related topics. Provide concise and practical advice on these areas, including specific strategies for selling various products. Format your response in markdown, using appropriate headers, lists, and emphasis."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=300,
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        logger.info("Successfully received chatbot response from Groq")
        return jsonify({"response": ai_response})
    except Exception as e:
        error_msg = f"Error getting chatbot response from Groq: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)