import React, { useState } from "react";
import axios from "axios";

const TransactionForm = ({ setBalance, setSavings, onTransaction }) => {
  const [amount, setAmount] = useState("");
  const [type, setType] = useState("luxurious");
  const [suggestion, setSuggestion] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://127.0.0.1:5002/transaction", {
        user_id: 1,
        amount: parseFloat(amount),
        type,
        feedback: true, // Assume user accepts by default
      });
      setSuggestion(response.data);
    } catch (error) {
      console.error("Error submitting transaction:", error);
    }
  };

  const handleAccept = () => {
    setBalance(suggestion.new_balance);
    setSavings(suggestion.new_savings);
    onTransaction(suggestion.new_balance, suggestion.new_savings); // Trigger graph update
    setSuggestion(null);
  };

  const handleReject = () => {
    setSuggestion(null);
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow-md">
      <h2 className="text-lg font-bold mb-2">New Transaction</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="number"
          className="border p-2 w-full mb-2"
          placeholder="Enter amount"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
        />
        <select className="border p-2 w-full mb-2" value={type} onChange={(e) => setType(e.target.value)}>
          <option value="luxurious">Luxurious</option>
          <option value="essential">Essential</option>
        </select>
        <button type="submit" className="bg-blue-500 text-white p-2 w-full">
          Submit
        </button>
      </form>

      {suggestion && (
        <div className="mt-4 p-3 bg-gray-200 rounded">
          <p>Suggested Savings: ${suggestion.savings_suggestion}</p>
          <p>New Total: ${suggestion.new_total}</p>
          <button className="bg-green-500 text-white p-2 m-2" onClick={handleAccept}>
            Accept
          </button>
          <button className="bg-red-500 text-white p-2" onClick={handleReject}>
            Reject
          </button>
        </div>
      )}
    </div>
  );
};

export default TransactionForm;
