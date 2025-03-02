import React, { useState } from "react";
import BalanceCard from "./components/BalanceCard";
import TransactionForm from "./components/TransactionForm";
import SavingsChart from "./components/SavingsChart";

function App() {
  const [balance, setBalance] = useState(5000);
  const [savings, setSavings] = useState(0);
  const [updateTrigger, setUpdateTrigger] = useState(0); // Used to force re-render of SavingsChart

  const handleTransactionUpdate = (newBalance, newSavings) => {
    setBalance(newBalance);
    setSavings(newSavings);
    setUpdateTrigger((prev) => prev + 1); // Change state to trigger graph update
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold text-center mb-4">Savings Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <BalanceCard balance={balance} savings={savings} />
        <TransactionForm setBalance={setBalance} setSavings={setSavings} onTransaction={handleTransactionUpdate} />
      </div>
      <div className="mt-6">
        <SavingsChart updateTrigger={updateTrigger} />
      </div>
    </div>
  );
}

export default App;
