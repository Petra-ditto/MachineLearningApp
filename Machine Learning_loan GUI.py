import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import traceback

# Load the trained model pipeline
MODEL_PATH = "/Users/petraszabolcsi/Documents/Excel/Models/logreg_pipeline.joblib"
model = joblib.load(MODEL_PATH)

# GUI App
class LoanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Loan Approval Prediction")
        self.root.geometry("420x550")
        self.root.resizable(False, False)

        title = ttk.Label(root, text="Credit Approval Predictor", font=("Arial", 18, "bold"))
        title.pack(pady=10)

        self.fields = {}

        form_frame = ttk.Frame(root, padding=10)
        form_frame.pack(fill="both", expand=True)

        # Fields (label + entry/combobox)
        inputs = {
            "Gender": ["Male", "Female"],
            "Married": ["Yes", "No"],
            "Dependents": ["0", "1", "2", "3+"],
            "Education": ["Graduate", "Not Graduate"],
            "Self_Employed": ["Yes", "No"],
            "Applicant_Income": None,
            "Coapplicant_Income": None,
            "Loan_Amount": None,
            "Loan_Amount_Term": ["360", "180", "120", "84", "60"],
            "Credit_History": ["1", "0"],
            "Property_Area": ["Urban", "Semiurban", "Rural"]
        }

        for label, options in inputs.items():
            frame = ttk.Frame(form_frame)
            frame.pack(fill="x", pady=5)

            ttk.Label(frame, text=label + ":").pack(side="left")

            if options is None:
                entry = ttk.Entry(frame)
                entry.pack(side="right", fill="x", expand=True)
                self.fields[label] = entry
            else:
                combo = ttk.Combobox(frame, values=options, state="readonly")
                combo.pack(side="right", fill="x", expand=True)
                self.fields[label] = combo

        # Predict button
        predict_btn = ttk.Button(root, text="Predict Approval", command=self.predict)
        predict_btn.pack(pady=15)

        # Output Label
        self.result_label = ttk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def predict(self):
        try:
            data = {}
            for label, widget in self.fields.items():
                value = widget.get()
                if value == "":
                    messagebox.showerror("Missing Data", f"Please enter {label}")
                    return

                # Convert numeric fields
                if label in ["Applicant_Income", "Coapplicant_Income", "Loan_Amount", "Loan_Amount_Term"]:
                    value = float(value)

                if label == "Dependents" and value == "3+":
                    value = "3+"

                data[label] = value

            df = pd.DataFrame([data])
            print("INPUT DATAFRAME:")
            print(df)

            pred = model.predict(df)[0]
            print("PREDICTION:", pred)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(df)[0][1]
            else:
                prob = None

            if pred == 1:
                verdict = f"✔ APPROVED (prob={prob:.2f})" if prob else "✔ APPROVED"
                color = "green"
            else:
                verdict = f"✘ REJECTED (prob={prob:.2f})" if prob else "✘ REJECTED"
                color = "red"

            self.result_label.config(text=verdict, foreground=color)

        except Exception as e:
            print("\nERROR OCCURRED:")
            traceback.print_exc()  # <-- shows full error in console
            messagebox.showerror("Error", str(e))


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = LoanApp(root)
    root.mainloop()
