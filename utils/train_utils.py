import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, trainloader, optimizer, criterion_flag, criterion_sub, epochs=10, scheduler=None):
    model.to(device)
    print(f"Number of model parameters: {count_parameters(model):,}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for inputs, flag_targets, sub_targets in trainloader:
            inputs = inputs.to(device)
            flag_targets = flag_targets.to(device)
            sub_targets = sub_targets.to(device)

            optimizer.zero_grad()
            flag_logits, sub_logits = model(inputs)
            loss_flag = criterion_flag(flag_logits, flag_targets)
            loss_sub = criterion_sub(sub_logits, sub_targets)
            loss = loss_flag + loss_sub
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        avg_loss = running_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")
    return model

def test_model(model, testloader):
    model.to(device)
    model.eval()
    print(f"Number of model parameters: {count_parameters(model):,}")
    correct_flag, correct_sub, total = 0, 0, 0
    with torch.no_grad():
        for inputs, flag_targets, sub_targets in testloader:
            inputs = inputs.to(device)
            flag_targets = flag_targets.to(device)
            sub_targets = sub_targets.to(device)
            flag_logits, sub_logits = model(inputs)
            _, flag_preds = flag_logits.max(1)
            _, sub_preds = sub_logits.max(1)
            total += flag_targets.size(0)
            correct_flag += flag_preds.eq(flag_targets).sum().item()
            correct_sub += sub_preds.eq(sub_targets).sum().item()
    flag_acc = 100. * correct_flag / total if total > 0 else 0.0
    sub_acc = 100. * correct_sub / total if total > 0 else 0.0
    print(f"Gluten flag accuracy: {flag_acc:.2f}%")
    print(f"Substitute class accuracy: {sub_acc:.2f}%")
    return flag_acc, sub_acc

