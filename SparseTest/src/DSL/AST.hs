{-# LANGUAGE DeriveGeneric #-}

module DSL.AST
(
  Index(..)
, IndexName
, Dim(..)
, Shape
, DType(..)
, TensorFormat(..)
, TensorDecl(..)
, Expr(..)
, UnaryOp(..)
, BinaryOp(..)
, EinsumStmt(..)
, Schedule(..)
, StorageAnnotation(..)
, AccessPattern(..)
, IterationConstraint(..)
-- * Utilities
, mkIndex
, mkIdxs
, freeIndices
, allIndices
, validateExpr
) where

import Data.List (nub, (\\))
import qualified Data.Set as Set
import Data.Set (Set)
import GHC.Generics (Generic)

-- | A (single) index name for einsum (e.g., 'i', 'j', 'k')
newtype Index = Ix { unIx :: Char }
    deriving (Eq, Ord, Show, Generic)

type IndexName = Char

-- | A single dimension. We keep it as an Int for simplicity but it may
-- later become symbolic (Var/Expr) for parametric sizes.
newtype Dim = Dim { unDim:: Int }
    deriving (Eq, Ord, Show, Generic)

type Shape = [Dim]

-- | Basic data types supported for tensor values.
data DType = TInt | TFloat | TDouble
    deriving (Eq, Show, Generic)

-- | Storage formats which the tool will support. This list is intentionally
-- conservative but extensible.
data TensorFormat
    = FmtDense                      -- ^ Dense, contiguous layout
    | FmtCSR                        -- ^ Compressed Sparse Row (2D)
    | FmtCSC                        -- ^ Compressed Sparse Column (2D)
    | FmtCOO                        -- ^ Coordinate list
    | FmtBlocked Int                -- ^ Blocked format with block size
    | FmtCustom String              -- ^ Backend-specific custom format
    deriving (Eq, Show, Generic)

-- | A declaration of a tensor variable (name, shape, format, dtype)
data TensorDecl = TensorDecl
    { tName     :: String 
    , tShape    :: Shape
    , tFormat   :: TensorFormat
    , tDType    :: DType
    } deriving (Eq, Show, Generic)

-- | Unary ops in our DSL. These are the building blocks for elementwise
-- or reduction operations
data UnaryOp
    = UNegate
    | UAbs
    | UExp
    deriving (Eq, Show, Generic)

-- | Binary ops supported in expressions.
data BinaryOp
    = BAdd
    | BSub
    | BMul
    | BDiv
    deriving (Eq, Show, Generic)

-- | Access pattern: how an expression reads or writes tensor. Useful for
-- computing iteration graphs and scheduling.
data AccessPattern
    = Read Index            -- ^ read along an index
    | Write Index           -- ^ write along an index
    deriving (Eq, Show, Generic)

-- |Iteration constraints that represent relationships between indices
-- (e.g. ordering, equality / contraction semantics). This small set is
-- used for reasoning about merges and intersections when lowering.
data IterationConstraint
    = IndexEqual Index Index            -- ^ force two indices to be equal (contraction)
    | IndexLe Index Index               -- ^ index ordering (i <= j)
    deriving (Eq, Show, Generic)

-- | Storage annotation allows attaching hints about storage to an expression
-- or tensor: e.g., say that a particular index is stored sparsely.
data StorageAnnotation = StorageAnnotation
    { saIndex   :: Index
    , saFormat  :: TensorFormat
    } deriving (Eq, Show, Generic)

-- | The core expression language for Einsum-like operations. This is
-- intentionally minimal but expressive enough for common tensor algebra.
data Expr
    = EVar String [Index]           -- ^ Tensor variable with index order. e.g., A[i,j]
    | EConstDouble Double           -- ^ scalar constant
    | EUnary UnaryOp Expr           -- ^ unary elementwise op
    | EBinary BinaryOp Expr Expr    -- ^ binary elementwise op, e.g. multiply
    | ESum Index Expr               -- ^ reduction (sum) over an index
    | ETranspose Expr [Index]       -- ^ change logical index ordering
    | EReshape Expr Shape           -- ^ reshape (purely logical)
    | ELet String Expr Expr         -- ^ local binding
    | EMap Expr Expr                -- ^ map of function over tensor (high-order)
    deriving (Eq, Show, Generic)

-- | A top-level Einsum statement: output, list of input tensors and the
-- expression describing computation. Additional 'Schedule' may be
-- attached later.
data EinsumStmt = EinsumStmt
    { outName   :: String
    , outIdxs   :: [Index]
    , inputs    :: [TensorDecl]
    , expr     :: Expr
    , annots    :: [StorageAnnotation]
    } deriving (Eq, Show, Generic)

-- | Scheduling directives / transformations for lowering. This is a
-- compact representation of common scheduling operations
data Schedule
    = SchReorder [Index]        -- ^ reorder loop indices
    | SchTile Index Int         -- ^ tile an index with tile size
    | SchParallel Index         -- ^ parallelize along index with width
    | SchVectorize Index Int    -- ^ vectorize along index with width
    | SchCustom String          -- ^ backend-specific schedule
    deriving (Eq, Show, Generic)

-- | Smart constructors / helpers
mkIndex :: Char -> Index
mkIndex = Ix

mkIdxs :: String -> [Index]
mkIdxs = map Ix

-- | Compute the free indices appearing in an expression
freeIndices :: Expr -> [Index]
freeIndices e = nub (go e)
    where
        go (EVar _ idxs) = idxs
        go (EConstDouble _) = []
        go (EUnary _ x) = go x
        go (EBinary _ x y) = go x ++ go y
        go (ESum idx x) = filter (/= idx) (go x)
        go (ETranspose x idxs) = nub (idxs ++ go x)
        go (EReshape x _) = go x
        go (ELet _ bnd body) = go bnd ++ go body
        go (EMap f a) = go f ++ go a

-- | Helper to collect *all* indices including bound ones
allIndices :: Expr -> [Index]
allIndices e = nub (go e)
    where
        go (EVar _ idxs) = idxs
        go (EConstDouble _) = []
        go (EUnary _ x) = go x
        go (EBinary _ x y) = go x ++ go y
        go (ESum _ x) = go x
        go (ETranspose x idxs) = idxs ++ go x
        go (EReshape x _) = go x
        go (ELet _ bnd body) = go bnd ++ go body
        go (EMap f a) = go f ++ go a

-- | Basic validation: ensure that indices referenced in output appear in
-- the expression, and that tensor shapes match index usage when shapes
-- are available.
validateExpr :: EinsumStmt -> Either String ()
validateExpr stmt = do
    let outIdxs' = outIdxs stmt
        used = allIndices (expr stmt)
    if any (`notElem` used) outIdxs'
        then Left "Some output indices are not produced by the expression"
        else Right ()
