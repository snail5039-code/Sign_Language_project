create table if not exists board(
    id bigserial primary key
    ,boardName varchar(100) not null
);

create table if not exists boards(
    id bigserial primary key
    ,title varchar(200) not null
    ,content text not null
    ,regDate timestamp not null default now()
    ,updateDate timestamp not null default now()
    ,boardId bigint not null
    ,memberId bigint not null
    
) 
INSERT INTO board (id, boardName) VALUES
  (1, '공지사항'),
  (2, '자유게시판'),
  (3, '질문과 답변'),
  (4, '오류사항 접수')
ON CONFLICT (id) DO NOTHING;