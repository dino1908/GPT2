import torch
from utils.helper_functions import calc_loss_batch, calc_loss_loader, generate, text_to_token_ids, token_ids_to_text

def train_model_simple(model, train_dataloader, val_dataloader, num_epochs, optimizer, 
                    tokenizer, eval_freq, eval_iter, start_context, device):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = calc_loss_batch(X, y, model, device=device)
            loss.backward()
            optimizer.step()
            tokens_seen += X.numel()
            global_step += 1

        
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                model, train_dataloader, val_dataloader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device=device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device=device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embed.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

#print("Model trainer functions success!")