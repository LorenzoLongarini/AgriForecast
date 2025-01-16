import xml.etree.ElementTree as ET
import csv

def xml_to_csv(xml_file, csv_file, root_element, row_element):
    try:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Check if the root element is correct
        if root.tag != root_element:
            raise ValueError(f"Expected root element '{root_element}', but got '{root.tag}'")

        # Extract rows
        rows = root.findall(row_element)
        if not rows:
            raise ValueError(f"No elements found with tag '{row_element}'")

        # Extract column names from the first row
        headers = [child.tag for child in rows[0]]

        # Write data to CSV
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write headers
            writer.writerow(headers)

            # Write rows
            for row in rows:
                writer.writerow([child.text for child in row])

        print(f"File converted successfully and saved as {csv_file}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
xml_file = ".\\assets\\Paolo_test_1.xml"

csv_file = "TSA.csv"
root_element = "DataPoints"  # Replace with the actual root element name
row_element = "DataPoint"    # Replace with the actual row element name

xml_to_csv(xml_file, csv_file, root_element, row_element)