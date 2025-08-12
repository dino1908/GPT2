import torch
import tiktoken
import time
from GPT_CONFIG import GPT_CONFIG
from GPT_Model_Class import GPTModel
from Data_Loader_class import create_dataloader_v1
from helper_functions import calc_loss_loader, generate, text_to_token_ids, token_ids_to_text
from model_trainer_functions import train_model_simple

#Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#Load Model & Tokenizer
model = GPTModel(GPT_CONFIG).to(device)
tokenizer = tiktoken.get_encoding("gpt2")

#Load file
f = "C:\\Users\\Roshan\Documents\\DL\\DL_modular\\the-verdict.txt"
with open(f, "r", encoding="utf-8") as file:
    text_data = file.read()
print(f"Book Loaded Successfully. Length of book is {len(text_data)}")

#Divide Test & Val data
train_ratio = 0.9
split = int(0.9*len(text_data))
train_data = text_data[:split]
val_data = text_data[split:]
print(f"Length of Train Data is {len(train_data)}")
print(f"Length of Val Data is {len(val_data)}")

#Creating Data Loaders
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=True,
    shuffle=True
 )

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=False,
    shuffle=False
 )

#Initial Loss
#initial_train_loss = calc_loss_loader(train_loader, model, device=device)
#initial_val_loss = calc_loss_loader(val_loader, model, device=device)
#print(f"Initial Train Loss: {initial_train_loss} | Initial Validation Loss: {initial_val_loss}")


#Training starts here!!

start_time = time.time()
device = device

model = GPTModel(GPT_CONFIG)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model, train_dataloader=train_loader, val_dataloader=val_loader, optimizer=optimizer, device=device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

print("Training completed successfully! Existing training block")

#Training Ends

#Evalauting results
model.to("cpu")
model.eval()

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG["context_length"]
)

print("A sample from trained model:\n")
print(token_ids_to_text(token_ids,tokenizer))

#Saving Model
torch.save({"model_state_dict":model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, "model.pth")

print("Model Saved Successfully")