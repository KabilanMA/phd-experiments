{-# LANGUAGE DeriveGeneric #-}

module DSL.AST
(
    Index(..),
    Dim(..),
    Shape,
    DType(..),
    LevelFormat(..),
    TensorFormat,
    BinaryOp(..),
    AccessPattern(..),
    TensorDecl(..),
    TensorStorage(..),
    Tensor(..),
    AxisIter(..),
    Expr(..),
    Stmt(..),
    EinsumOp(..)
) where

import GHC.Generics (Generic)

-- =========================================
-- Basic Types
-- =========================================

-- | A (single) index name for einsum (e.g., 'i', 'j', 'k')
newtype Index = Index { unIx :: Char }
    deriving (Eq, Ord, Show, Generic)

-- | An integer value to indicate the dimension of each tensor axis
newtype Dim = Dim { unDim:: Int }
    deriving (Eq, Ord, Show, Generic)

-- | Shape of a tensor
type Shape = [Dim]

-- Basic data types supported for tensor values.
data DType = TInt | TFloat | TDouble
    deriving (Eq, Show, Generic)

-- | Each dimension can be Dense or Sparse
data LevelFormat = Dense | Sparse
    deriving (Eq, Show)

type TensorFormat = [LevelFormat]

-- | Binary Operations
data BinaryOp = BAdd | BSub | BMul | BDiv
    deriving (Eq, Show, Generic)

-- | Access pattern: whether an expression reads or writes a tensor
data AccessPattern = Read | Write
    deriving (Eq, Show, Generic)


-- =========================================
-- Tensor Declarations & Storage
-- =========================================

-- | A declaration of a tensor variable (name, shape, tensor format, dtype)
data TensorDecl = TensorDecl
    { tName         :: String
    , tShape        :: Shape
    , tFormat       :: TensorFormat
    , tDType        :: DType
    } deriving (Eq, Show, Generic)

-- | Low-level storage information for sparse tensors
data TensorStorage = TensorStorage
    { tsValues      :: String       -- name of data array
    , tsPtrs        :: [String]     -- pointer arrays for sparse axes
    , tsIndices     :: [String]     -- index ararys for sparse axes
    } deriving (Eq, Show, Generic)

-- | A complete tensor, linking declaration to storage
data Tensor = Tensor
    { decl      :: TensorDecl
    , storage   :: TensorStorage
    } deriving (Eq, Show, Generic)

-- =========================================
-- Axis Iterators
-- =========================================

-- | Type of axis iteration (dense, compressed, hashed, etc.)
data AxisIter
    = DenseIter Index Dim           -- for dense axes
    | CompressedIter Index Dim TensorStorage Int    -- compressed axis (e.g., CSR/CSC)
    | HashedIter Index Dim TensorStorage    -- hashed axis
    deriving (Eq, Show, Generic)

-- =========================================
-- Expressions
-- =========================================

data Expr
    = TensorVar Tensor              -- reference to a tensor
    | Const Double                  -- scalar constants
    | BinOp BinaryOp Expr Expr      -- e.g., Add, Mul, Sub, Div
    | Sum Index Expr                -- reduction / summation over an index
    | IndexAt Tensor [Index] AccessPattern  -- indexing into tensor with read/write info
    deriving (Eq, Show, Generic)

-- =========================================
-- Statements
-- =========================================

data Stmt
    = Assign String Expr            -- e.g., C = A + B
    | Loop AxisIter Stmt            -- loop over dense or sparse axis
    | Seq [Stmt]                    -- sequence of statements
    deriving (Eq, Show, Generic)

-- =========================================
-- High-level Operations
-- =========================================

-- | Representation of an einsum / tensor contraction
data EinsumOp = EinsumOp
    { output    :: TensorDecl       -- output tensor
    , inputs    :: [TensorDecl]     -- input tensors
    , indices   :: [[Index]]        -- indices per tensor
    , reduction :: [Index]          -- reduction indices
    } deriving (Eq, Show, Generic)

-- pretty print einsum
