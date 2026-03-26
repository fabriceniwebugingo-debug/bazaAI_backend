-- Create conversation_state table if it doesn't exist
CREATE TABLE IF NOT EXISTS conversation_state (
    phone VARCHAR(20) PRIMARY KEY,
    last_type VARCHAR(50),
    options TEXT,
    updated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for better performance
CREATE INDEX IF NOT EXISTS idx_conversation_state_updated_at ON conversation_state(updated_at);

-- Create airtime_transfers table if it doesn't exist (for fraud detection)
CREATE TABLE IF NOT EXISTS airtime_transfers (
    id SERIAL PRIMARY KEY,
    from_phone VARCHAR(20) NOT NULL,
    to_phone VARCHAR(20) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'completed'
);

-- Create indexes for airtime_transfers
CREATE INDEX IF NOT EXISTS idx_airtime_transfers_from_phone ON airtime_transfers(from_phone);
CREATE INDEX IF NOT EXISTS idx_airtime_transfers_created_at ON airtime_transfers(created_at);
