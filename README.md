# Long-Term User Embedding Model for Personalized Content Recommendations

## Overview
This project focuses on building a long-term user embedding model for personalized content recommendations. The system processes large-scale catalog data stored in Google Cloud Storage (GCS), with a distributed training setup utilizing multiple GPUs. The goal is to generate user embeddings to drive personalized recommendations.

## Key Components
1. **Data Pipeline**:
   - Retrieving user journey files in Parquet format from Google Cloud Storage.
   - Splitting the data into training and validation sets, ensuring a balanced distribution across GPUs.
   - Constructing vocab indices for categorical features (event_type, catalog_id, sscat_id).

2. **Model Architecture**:
   - **PinnerFormer**: A neural network architecture that includes:
     - **UserTower**: Encodes user behavior sequences.
     - **CatalogTower**: Encodes catalog item representations.
     - **CatalogEmbedding**: Embeds catalog items for efficient retrieval.

3. **Training Setup**:
   - Distributed training using PyTorch Lightning on multiple GPUs (NVIDIA L4 devices).
   - Utilizes mixed-precision (fp16) for training efficiency.
   - Cloud logging and checkpointing to ensure training reproducibility and manageability.

4. **Training**:
   - The model was trained with configurable batch size, number of epochs, and validation frequency, ensuring efficient use of resources and time.

## Results
- Successfully launched distributed training using GPUs.
- Achieved reproducible, scalable training with 80M trainable parameters.
- The resulting model is capable of generating user embeddings for downstream recommendation tasks.


