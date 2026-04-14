import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_logs(log_file):

    iter_steps, train_loss = [], []
    val_steps, val_loss = [], []

    #Dateipfad mit pathlib/Path "importieren" => Dateipfad nicht mehr hardcoded.
    p = Path(__file__).parents[1] /"nanoGPT" /"logs" / log_file
    print("Path:", p)

    #Dokument importieren, text parsen, nach pattern schauen und printen
    #Zwei leere listen füllen. Eine liste für training/loss und die andere für validation/loss => iter_steps/train_loss, val_step/val_loss. 
    #Nur das Validation match is 3 werte lang, bei den "normalen" training output sind es 2 werte.
    with open(p, encoding="utf-16") as f:
        for line in f:
            match = re.findall(r"iter.(\d\d+):.loss.(\d.\d+)", line) or re.findall(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", line)
            if match:
                if len(match[0]) == 2:
                    iter_steps.append(float(match[0] [0]))
                    train_loss.append(float(match[0] [1]))
                if len(match[0]) == 3:
                    val_steps.append(float(match[0] [0]))
                    val_loss.append(float(match[0] [2]))


    #print("Training_Values:", iter_steps, train_loss)
    #print("Validation_Values:", val_steps, val_loss)
    return (iter_steps, train_loss)

### Plotting ###


baseline_steps, baseline_loss = parse_logs("baseline.log")
lowLR_steps, lowLR_loss = parse_logs("lowLR.log")

plt.plot(baseline_steps, baseline_loss, label= "Baseline")
plt.plot(lowLR_steps, lowLR_loss, label= "Low Learning Rate (1e-4)")
plt.title("NanoGPT_Training")
plt.xlabel("Itteration Step")
plt.ylabel("Training Loss")
plt.grid(alpha = 0.4)
plt.xlim(0)
plt.legend()

#plt.show()
plt.savefig("plot.png")