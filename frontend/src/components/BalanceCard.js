import React from "react";

const BalanceCard = ({ balance, savings }) => {
  return (
    <div className="bg-blue-200 p-4 rounded-lg shadow-md text-center">
      <h2 className="text-xl font-bold">Current Balance</h2>
      <p className="text-2xl font-semibold">${balance.toFixed(2)}</p>
      <h3 className="text-lg font-bold mt-2">Total Savings</h3>
      <p className="text-xl font-semibold">${savings.toFixed(2)}</p>
    </div>
  );
};

export default BalanceCard;
