import React, { useEffect, useState } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

const SavingsChart = ({ updateTrigger }) => {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await axios.get("http://127.0.0.1:5002/savings_trend?user_id=1");
        const data = response.data;
        setChartData({
          labels: data.map((entry) => entry.transaction_number),
          datasets: [
            { label: "Balance", data: data.map((entry) => entry.balance), borderColor: "blue", fill: false },
            { label: "Savings Amount", data: data.map((entry) => entry.savings_amount), borderColor: "green", fill: false },
            { label: "Savings Percentage", data: data.map((entry) => entry.savings_percentage), borderColor: "red", fill: false },
          ],
        });
      } catch (error) {
        console.error("Error fetching savings trend:", error);
      }
    }

    fetchData(); // Fetch immediately on mount
    const interval = setInterval(fetchData, 3000); // Poll every 3 seconds

    return () => clearInterval(interval); // Cleanup interval on unmount
  }, [updateTrigger]); // Refetch data when updateTrigger changes

  return chartData ? <Line data={chartData} /> : <p>Loading...</p>;
};

export default SavingsChart;
