# MTG Card Reader

A full-stack application that analyzes Magic: The Gathering card images and returns real-time pricing information. Upload a card photo, and the app will identify the card and display its current market price.

## Features

- **Image Processing**: Removes backgrounds and automatically rotates cards to the correct orientation
- **OCR**: Extracts card titles from processed images using custom-trained models
- **Card Lookup**: Queries the Scryfall API to fetch accurate card data and pricing
- **Responsive UI**: Modern, user-friendly interface built with Next.js and Shadcn UI
- **Real-time Results**: Displays rarity, regular price, and foil price information

## Tech Stack

### Frontend
- **Next.js** - React framework for production applications
- **TypeScript** - Type-safe JavaScript
- **Shadcn UI** - High-quality React component library
- **TailwindCSS** - Utility-first CSS framework
- **React Hook Form** - Performant, flexible form validation

### Backend
- **Flask** - Python web framework
- **Python** - Core processing logic
- **Scryfall API** - MTG card data source

### ML Models
- **Background Remover** - U2Net model for removing card backgrounds
- **Image Rotator** - Classifier for detecting and correcting card orientation
- **Title Extractor** - Fine-tuned OCR model trained on MTG card fonts

## Getting Started

### Prerequisites

- Node.js 16+ and npm/pnpm
- Python 3.8+
- pip (Python package manager)

### Frontend Setup

1. Install dependencies:
   ```bash
   pnpm install
   # or
   npm install
   ```

2. Start the development server:
   ```bash
   pnpm dev
   # or
   npm run dev
   ```

   The frontend will be available at `http://localhost:3000`

### Backend Setup

1. Navigate to the server-backend directory:
   ```bash
   cd server-backend
   ```

2. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required model weights:

   - **Background Remover & Image Rotator Models**: Download from [Google Drive](https://drive.google.com/drive/folders/1Ew0PMWjVed00vFgXnD6B__0HgQ5iVAcN?usp=sharing)
     - Extract the model files to `server-backend/backend/modules/background_remove/model_weights/` and `server-backend/backend/modules/image_preprocess/model_weights/`

   - **Title Extractor Model**: Download from [HuggingFace](https://huggingface.co/ben451/tocr-small-microsoft-mtg-card-font/settings)
     - Extract the model files to `server-backend/backend/modules/title_extract/`

5. Start the Flask backend:
   ```bash
   python app.py
   ```

   The backend will be running at `http://localhost:5000`

### Running the Full Application

1. Ensure the Flask backend is running on `http://localhost:5000`
2. In a separate terminal, run the Next.js frontend on `http://localhost:3000`
3. Open your browser and navigate to `http://localhost:3000`
4. Upload a Magic: The Gathering card image and get pricing information

## File Structure

```
├── app/                              # Next.js app directory
│   ├── api/
│   │   └── analyze/
│   │       └── route.ts             # API endpoint for image analysis
│   ├── layout.tsx                   # Root layout component
│   ├── page.tsx                     # Main page component
│   └── globals.css                  # Global styles
│
├── components/                      # React components
│   ├── upload-page.tsx              # File upload interface
│   ├── loading-page.tsx             # Loading state component
│   ├── results-page.tsx             # Results display component
│   ├── theme-provider.tsx           # Theme context provider
│   └── ui/                          # Shadcn UI components
│
├── hooks/                           # Custom React hooks
│   ├── use-toast.ts                 # Toast notification hook
│   └── use-mobile.ts                # Mobile detection hook
│
├── lib/                             # Utility functions
│   └── utils.ts                     # General utilities
│
├── server-backend/                  # Flask backend
│   ├── app.py                       # Flask application entry point
│   ├── requirements.txt             # Python dependencies
│   ├── backend/
│   │   └── modules/
│   │       ├── scryfall_card_finder.py     # Scryfall API integration
│   │       ├── background_remove/
│   │       │   ├── background_remover.py   # Background removal logic
│   │       │   ├── u2net.py               # U2Net model architecture
│   │       │   └── model_weights/         # Model weights (downloaded separately)
│   │       ├── image_preprocess/
│   │       │   ├── image_preprocessor.py  # Image preprocessing & rotation
│   │       │   └── model_weights/         # Model weights (downloaded separately)
│   │       └── title_extract/
│   │           ├── title_extractor.py     # OCR logic
│   │           └── [HuggingFace model files]
│   └── model_setups/                # Model training & development
│
├── public/                          # Static assets
├── styles/                          # Global styling
├── package.json                     # Node.js dependencies
├── tsconfig.json                    # TypeScript configuration
├── next.config.mjs                  # Next.js configuration
└── README.md                        # This file
```

## How It Works

1. **Upload**: User uploads a Magic: The Gathering card image via the web interface
2. **Background Removal**: The image is processed through the U2Net model to remove the background
3. **Orientation Detection**: The image is analyzed and rotated to the correct orientation if needed
4. **Text Extraction**: OCR model extracts the card title from the processed image
5. **Card Lookup**: The extracted title is used to query the Scryfall API
6. **Price Display**: Card data including rarity and pricing is displayed to the user

## API Endpoints

### POST `/api/fetch_price`
Analyzes an uploaded card image and returns pricing information.

**Request:**
```
multipart/form-data
- image: File
```

**Response:**
```json
{
  "status": "success",
  "rarity": "rare",
  "price": "$5.25",
  "price_foil": "$12.50",
  "collector_num": "123"
}
```

## Environment Variables

Create a `.env.local` file in the root directory if needed for frontend configuration.

## Development

### Build
```bash
pnpm build
# or
npm run build
```

### Linting
```bash
pnpm lint
# or
npm run lint
```

## License

This project is provided as-is for personal and educational use.

## Troubleshooting

- **Model files not found**: Ensure you've downloaded the model weights from the provided links and extracted them to the correct directories
- **CORS errors**: Ensure Flask backend is running with CORS enabled
- **Card not recognized**: Try uploading a clearer, well-lit image of the card
- **Port conflicts**: If ports 3000 or 5000 are in use, update the configuration accordingly

## Future Enhancements

- Support for multiple card variants
- Bulk card analysis
- Price history tracking
- Mobile app version
- Card collection management
