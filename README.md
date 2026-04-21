# AI Purchase Cost Project

## Dataset Overview

The dataset (`AI purchase cost project (STUDENT DATA) V2 (1)(Blad2).csv`) contains historical data of shipments and their corresponding configurations and pricing classifications. 

The goal (inferred) is to predict the **Price category** or understand the key drivers behind the pricing of shipments, using features such as payload (Payweight) and varying logistical requirements.

## Data Dictionary

| Feature | Description |
| :--- | :--- |
| **Shipment** | Unique shipment reference number. |
| **LAADDATUM** | Loading date of the shipment. |
| **Load code** | Code indicating the loading location. |
| **Unload code** | Code indicating the delivery/unloading location. |
| **Payweight** | Chargeable weight used for pricing. This is always the highest value between loading meters, volume, or kilograms. |
| **Distribution driven by code** | Indicates the logistics partner used for the distribution. |
| **Price category** | Pricing classification applied to the shipment *(Likely the target variable)*. |
| **Crossdock** | Indicates whether the shipment is handled via cross-docking. |
| **ADR** | Specifies if the shipment contains dangerous goods (ADR). |
| **Express** | Indicates if the shipment is handled as an express delivery. |
| **Thermo** | Indicates if temperature-controlled transport is required. |

## Example Shipment Computation

Here is an example to illustrate how features come together, particularly the `Payweight` calculation:

* **Shipment:** 2261707
* **LAADDATUM:** 1-1-2023
* **Load code:** 217 *(e.g., NL-10)*
* **Unload code:** 535 *(e.g., NL-55)*
* **Payweight:** 647.5
  * *Calculation Breakdown:* The physical KG is 500, volume is 2.1, and loading meters are 0.37. Based on the payweight formulas (volume * 330, loading meters * 1750, and actual KG), the highest value defines the Payweight.
* **Distribution driven by code:** 19 *(e.g., KLG Eersel)*
* **Price category:** C *(e.g., 500-600 range)*
