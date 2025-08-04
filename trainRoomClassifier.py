import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import random


with open('roomLabels.txt', 'r') as f:
    room_labels = set(line.strip().upper() for line in f if line.strip())

non_room_samples = [
    "Janitors Closet", "Main Entrance", "Library", "Elevator", "Office", "Hallway",
    "Storage", "Reception", "Lobby", "Conference Room", "Restroom", "Cafeteria", "Recycle", "BA104 ->", "Sidney Smith", "MaRS", "Termerty",
    
    "Turn Left", "Turn Right", "Go Straight", "Continue Down Hall", "End of Hallway",
    "Follow Signs", "Next Building", "Across the Street", "Up the Stairs", "Down the Hall",
    "Around the Corner", "Through the Door", "Past the Elevator", "Near the Exit",
    "Take Elevator", "Use Stairs", "Main Path", "Side Entrance", "Back Door",
    
    "Exit", "Emergency Exit", "Fire Exit", "Side Exit", "Back Exit", "North Exit",
    "South Exit", "East Exit", "West Exit", "Main Door", "Front Door", "Side Door",
    "Automatic Door", "Glass Door", "Security Door", "Access Door",
    
    "Stairs", "Stairwell", "Staircase", "Fire Stairs", "Emergency Stairs",
    "Elevator A", "Elevator B", "Freight Elevator", "Service Elevator", "Passenger Elevator",
    "Elevator Bank", "Lift", "Escalator", "Moving Walkway",
    
    "Mechanical Room", "Boiler Room", "Electrical Room", "Generator Room", "HVAC Room",
    "Utility Room", "Utility Closet", "Storage Room", "Supply Closet", "Cleaning Closet",
    "Janitorial Supply", "Maintenance Room", "Equipment Room", "Server Room", "IT Closet",
    "Telecom Room", "Network Closet", "Panel Room", "Meter Room",
    
    "Mens Restroom", "Womens Restroom", "Family Restroom", "Accessible Restroom",
    "Public Restroom", "Staff Restroom", "Water Fountain", "Drinking Fountain",
    "Bottle Filling Station", "Hand Sanitizer", "Paper Towel Dispenser",
    
    "Lobby Area", "Waiting Area", "Seating Area", "Common Area", "Student Lounge",
    "Study Area", "Quiet Zone", "Group Study", "Break Room", "Kitchen Area",
    "Dining Area", "Food Court", "Snack Bar", "Vending Area", "Microwave Station",
    
    "Information Desk", "Help Desk", "Reception Desk", "Front Desk", "Check-in",
    "Security Desk", "Guard Station", "Lost and Found", "First Aid Station",
    "Nurse Station", "Medical Office", "Counseling Center", "Student Services",
    "Admissions Office", "Registrar", "Financial Aid", "Bursar", "Cashier",
    
    "Computer Lab", "Printer Station", "Copy Center", "Copier", "Scanner",
    "Fax Machine", "ATM", "Card Reader", "Access Panel", "Intercom System",
    "Emergency Phone", "Fire Alarm", "Security Camera", "Surveillance",
    
    "Parking Lot", "Parking Garage", "Visitor Parking", "Staff Parking",
    "Courtyard", "Quad", "Garden", "Patio", "Terrace", "Balcony",
    "Sidewalk", "Walkway", "Path", "Trail", "Plaza", "Green Space",
    "Picnic Area", "Outdoor Seating", "Bike Rack", "Bus Stop",
    
    "North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest",
    "Left", "Right", "Straight", "Forward", "Back", "Up", "Down", "Above", "Below",
    "Next to", "Across from", "Behind", "In front of", "Adjacent to", "Nearby",
    
    "Open 9-5", "Closed", "After Hours", "Weekend Hours", "Holiday Hours",
    "By Appointment", "Staff Only", "Authorized Personnel", "Restricted Access",
    "Visitors Welcome", "Public Access", "Private", "No Entry", "Do Not Enter",
    
    "Meeting Room", "Conference Hall", "Assembly Hall", "Auditorium", "Theater",
    "Gymnasium", "Sports Center", "Fitness Center", "Pool", "Track",
    "Art Gallery", "Museum", "Exhibition Hall", "Display Case", "Bulletin Board",
    
    "Area", "Zone", "Section", "Department", "Division", "Unit", "Wing", "Block",
    "Complex", "Center", "Facility", "Location", "Destination", "Point", "Spot", "Place",
    
    "Under Construction", "Under Renovation", "Closed for Maintenance", "Out of Order",
    "Temporarily Closed", "Service Required", "Cleaning in Progress", "Wet Floor",
    "Caution", "Warning", "Danger", "Authorized Personnel Only",
    
    "WiFi Available", "Charging Station", "Power Outlet", "USB Port", "Network Access",
    "Internet Access", "Computer Terminal", "Kiosk", "Digital Display", "Monitor",
    
    "Fire Extinguisher", "Emergency Equipment", "AED", "First Aid Kit", "Safety Equipment",
    "Security Station", "Checkpoint", "Badge Reader", "Keypad", "Access Control",
    "Surveillance Area", "Monitored Zone", "Secure Area", "Restricted Zone",
    
    "Directory", "Map", "You Are Here", "Building Guide", "Floor Plan", "Legend",
    "Information Board", "Sign", "Marker", "Indicator", "Arrow", "Pointer",
    
    "Mail Room", "Package Pickup", "Delivery", "Shipping", "Receiving",
    "Loading Dock", "Service Entrance", "Vendor Entrance", "Delivery Entrance",
    
    "Wheelchair Access", "Accessible Route", "Ramp", "Accessible Parking",
    "Braille Sign", "Audio Assistance", "Visual Aid", "Hearing Loop",
    
    "Temperature Control", "Lighting Control", "Climate Control", "Ventilation",
    "Air Conditioning", "Heating", "Cooling", "Fresh Air", "Exhaust",
    "No Smoking", "Smoke Free", "Quiet Please", "Cell Phone Restriction",
    "Food Not Allowed", "Drink Not Allowed", "Pets Not Allowed",
    
    "Academic Advising", "Tutoring Services", "Study Skills", "Writing Center",
    "Math Help", "Language Lab", "Testing Center", "Proctoring Services",
    
    "Facility", "Service", "Resource", "Support", "Assistance", "Help Available",
    "Information Available", "Inquiries", "Questions", "Contact", "Connect",
    "Available", "Open", "Accessible", "Public", "Community", "Shared",

    "WI2006 ->",
    "UC140 ->",
    "<- SS2117", 
    "RU740 ->",
    "OIC154 ->",
    "NL6 ->",
    "<- MS2172",
    "IN313E ->",
        
    "WI2000-2010 ->",
    "WI520-530 ->",
    "WO20-40 ->",
    "WW115-125 ->",
    "WI2001-2020",
    "WO21-35",
    "WW120-130",
        
    "WI Building Directory",
    "WO Section Map",
    "WW Wing Guide",
    "WI Room Locations",
    "WO Room Guide", 
    "WW Floor Plan",
    
    "WI2001-2010",
    "KN520-530", 
    "IB20-40",
    "KK115-130",
]


data = []

for room in room_labels:
    data.append((room, 1))

for neg in non_room_samples:
    data.append((neg.upper(), 0))

random.shuffle(data)

train_val, test = train_test_split(data, test_size=0.4, random_state=42)
train, val = train_test_split(train_val, test_size=0.5, random_state=42)

print(f"Train samples: {len(train)}, Val samples: {len(val)}, Test samples: {len(test)}")

class RoomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=16):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class RoomClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output).squeeze(-1)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = RoomDataset(train, tokenizer)
val_dataset = RoomDataset(val, tokenizer)
test_dataset = RoomDataset(test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RoomClassifier().to(device)
criterion = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def eval_epoch(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss = train_epoch()
    val_loss, val_acc = eval_epoch(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

test_loss, test_acc = eval_epoch(test_loader)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

torch.save(model.state_dict(), 'room_classifier.pt')
tokenizer.save_pretrained('tokenizer/')