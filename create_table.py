import os
import json

def analyze_results_and_generate_table(results_folder, output_table_file):
    sensor_data = {}
    TDP = 55  # TDP del processore in watt
    emission_factor = 0.256  # kg CO₂ per kWh in Italia

    def format_number(num):
        """Formatta un numero in notazione scientifica se molto piccolo"""
        if num < 1e-3 and num != 0:
            return f"{num:.1e}"
        return f"{num:.3f}"

    for root, dirs, files in os.walk(results_folder):
        for file in files:
            if file.endswith("results.json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    overall = data.get("overall", {})
                    
                    # Determina il modello e il sensore dal percorso
                    relative_path = os.path.relpath(root, results_folder)
                    parts = relative_path.split(os.sep)
                    if len(parts) < 2:
                        continue  # Salta percorsi che non corrispondono al formato atteso
                    model = parts[0]
                    sensor_id = parts[1].replace("_merged", "")  # Rimuove "merged" dal nome del sensore

                    if model not in sensor_data:
                        sensor_data[model] = {}

                    if sensor_id not in sensor_data[model]:
                        sensor_data[model][sensor_id] = {
                            "mae_values": [],
                            "rmse_values": [],
                            "train_time": [],
                            "train_cpu": [],
                            "test_time": [],
                            "test_cpu": []
                        }

                    # Aggiungi i valori
                    sensor_data[model][sensor_id]["mae_values"].append(overall.get("MAE", float("inf")))
                    sensor_data[model][sensor_id]["rmse_values"].append(overall.get("RMSE", float("inf")))
                    sensor_data[model][sensor_id]["train_time"].append(overall.get("train_time", float("inf")))
                    sensor_data[model][sensor_id]["train_cpu"].append(overall.get("train_cpu", float("inf")))
                    sensor_data[model][sensor_id]["test_time"].append(overall.get("test_time", float("inf")))
                    sensor_data[model][sensor_id]["test_cpu"].append(overall.get("test_cpu", float("inf")))

    # Helper function to calculate average
    def safe_average(values):
        return sum(values) / len(values) if values else None

    # Genera i file della tabella per ciascun modello
    for model, sensors in sensor_data.items():
        table_rows = []
        all_mae = []
        all_rmse = []
        all_train_time = []
        all_train_cpu = []
        all_test_time = []
        all_test_cpu = []
        all_co2_train = []
        all_co2_test = []

        for sensor_id, data in sensors.items():
            avg_mae = safe_average(data["mae_values"])
            avg_rmse = safe_average(data["rmse_values"])
            avg_train_time = safe_average(data["train_time"])
            avg_train_cpu = safe_average(data["train_cpu"])
            avg_test_time = safe_average(data["test_time"])
            avg_test_cpu = safe_average(data["test_cpu"])

            # Calcola energia consumata e emissioni di CO₂
            train_energy = (avg_train_time * avg_train_cpu / 100 * TDP) / 3600  # kWh
            test_energy = (avg_test_time * avg_test_cpu / 100 * TDP) / 3600  # kWh
            co2_train = train_energy * emission_factor  # kg
            co2_test = test_energy * emission_factor  # kg

            # Colleziona per la media totale
            all_mae.append(avg_mae)
            all_rmse.append(avg_rmse)
            all_train_time.append(avg_train_time)
            all_train_cpu.append(avg_train_cpu)
            all_test_time.append(avg_test_time)
            all_test_cpu.append(avg_test_cpu)
            all_co2_train.append(co2_train)
            all_co2_test.append(co2_test)

            # Aggiungi i dati alla tabella
            table_rows.append(
                f"{sensor_id} & {format_number(avg_mae)} & {format_number(avg_rmse)} & {format_number(avg_train_time)} & {format_number(avg_train_cpu)} & {format_number(avg_test_time)} & {format_number(avg_test_cpu)} & {format_number(co2_train)} & {format_number(co2_test)} \\\\"
            )

        # Calcola i valori medi totali
        total_row = (
            "Totale & "
            f"{format_number(safe_average(all_mae))} & "
            f"{format_number(safe_average(all_rmse))} & "
            f"{format_number(safe_average(all_train_time))} & "
            f"{format_number(safe_average(all_train_cpu))} & "
            f"{format_number(safe_average(all_test_time))} & "
            f"{format_number(safe_average(all_test_cpu))} & "
            f"{format_number(safe_average(all_co2_train))} & "
            f"{format_number(safe_average(all_co2_test))} \\\\"
        )
        table_rows.append(total_row)

        # Scrivi la tabella LaTeX per il modello
        model_table_file = os.path.join(output_table_file, f"{model}_table.tex")
        with open(model_table_file, "w") as table_file:
            table_file.write("\\begin{table}[!h] \\centering \\rowcolors{2}{gray!25}{} \\begin{tabular}{p{3.2cm}p{0.9cm}p{0.9cm}p{0.9cm}p{1.2cm}p{0.9cm}p{1.2cm}p{1.2cm}p{1.2cm}}\n")
            table_file.write("\\toprule\n")
            table_file.write("\\textbf{Sensore} & $\\overline{\\text{MAE}}$ & $\\overline{\\text{RMSE}}$ & $\\overline{T}_{\\text{train}}$ (s) & $\\overline{U}_{\\text{CPU, train}}$ (\\%) & $\\overline{T}_{\\text{test}}$ (s) & $\\overline{U}_{\\text{CPU, test}}$ (\\%) & $E_{\\text{CO}_2, train}$ (kg) & $E_{\\text{CO}_2, test}$ (kg) \\\\\n")
            table_file.write("\\midrule\n")
            table_file.write("\n".join(table_rows))
            table_file.write("\\\\\n\\bottomrule \\end{tabular} \\caption{Valori medi di MAE, RMSE, e consumi per il modello " + model + "} \\label{tab:" + model + "_sensors_features} \\end{table}")

# Esegui lo script
# analyze_results_and_generate_table("path/to/results_folder", "path/to/output_folder")


analyze_results_and_generate_table("C:\\Users\\lollo\\Universita\\Tesi\\progetti\\AgriForecast\\assets\\results", "C:\\Users\\lollo\\Universita\\Tesi\\progetti\\AgriForecast\\assets\\results")
