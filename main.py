from transformers import AutoTokenizer, AutoModel

import torch

import torch.nn as nn

import torch.optim as optim



# ====================================

# 1. Dataset

# ====================================



documents = [



    "deep learning and neural networks for image classification",



    "machine learning model for disease prediction",



    "stock market analysis and financial forecasting",



    "investment strategies for financial growth",



    "online learning platforms for students",



    "new teaching methods in education"



]



labels = [



    0,  # AI

    0,  # AI



    1,  # Finance

    1,  # Finance



    2,  # Education

    2   # Education



]



label_names = {



    0: "AI",

    1: "Finance",

    2: "Education"



}



# ====================================

# 2. Load BERT

# ==================================== 



tokenizer = AutoTokenizer.from_pretrained(#Sentence↓Token IDs 

    "bert-base-uncased"

)



bert = AutoModel.from_pretrained(

    "bert-base-uncased"

)



# ====================================

# 3. Document Classifier

# ====================================



class DocumentClassifier(nn.Module):



    def __init__(self):



        super().__init__()



        self.bert = bert



        self.fc = nn.Linear(

            768,

            3

        )



    def forward(

        self,

        input_ids,

        attention_mask

    ):



        outputs = self.bert(



            input_ids=input_ids,



            attention_mask=attention_mask



        )



        cls_embedding = outputs.last_hidden_state[

            :,

            0,

            :

        ]



        logits = self.fc(

            cls_embedding

        )



        return logits



# ====================================

# 4. Model

# ====================================



model = DocumentClassifier()



criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(

    model.parameters(),

    lr=0.00001

)



# ====================================

# 5. Training

# ====================================



epochs = 10



for epoch in range(epochs):



    total_loss = 0



    for text, label in zip(

        documents,

        labels

    ):



        inputs = tokenizer(



            text,



            return_tensors="pt",



            truncation=True,



            padding=True,



            max_length=128



        )



        output = model(



            inputs["input_ids"],



            inputs["attention_mask"]



        )



        target = torch.tensor(

            [label]

        )



        loss = criterion(

            output,

            target

        )



        optimizer.zero_grad()



        loss.backward()



        optimizer.step()



        total_loss += loss.item()



    print(

        f"Epoch {epoch+1} "

        f"Loss: {total_loss:.4f}"

    )



# ====================================

# 6. Prediction Function

# ====================================



def predict_document(text):



    inputs = tokenizer(



        text,



        return_tensors="pt",



        truncation=True,



        padding=True,



        max_length=128



    )



    with torch.no_grad():



        output = model(



            inputs["input_ids"],



            inputs["attention_mask"]



        )



        prediction = torch.argmax(

            output,

            dim=1

        ).item()



    print("\nDocument:")

    print(text)



    print(

        "\nCategory:"

    )



    print(

        label_names[

            prediction

        ]

    )



# ====================================

# 7. Testing

# ====================================



predict_document(

    "deep learning for computer vision"

)



predict_document(

    "stock market investment analysis"

)



predict_document(

    "modern teaching techniques"

)
